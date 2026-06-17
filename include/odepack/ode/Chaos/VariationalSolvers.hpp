#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP

#include "../Core/Events.hpp"
#include "../OdeInt.hpp"
#include "../Core/VirtualBase.hpp"

namespace ode::chaos {

// ============================================================================
// DECLARATIONS
// ============================================================================


namespace detail{

// out (size 2*nsys) and in (size 2*nsys) can be the same pointer
template<typename T>
void normalized(T* out, const T* src, size_t nsys);

} // namespace detail

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class VariationalSolver;

template<typename T, size_t N, hasRhsFunc<T> OdeType>
pbox::owner<OdeRichSolver<T, 2*N>> get_virtual_variational_solver(Integrator method, T period, OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}, EVENTS events = {}){
    switch (method){
        case Integrator::RK4:
            return pbox::make_box<VariationalSolver<RK4, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        case Integrator::DOP853:
            return pbox::make_box<VariationalSolver<DOP853, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        case Integrator::RK23:
            return pbox::make_box<VariationalSolver<RK23, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        case Integrator::RK45:
            return pbox::make_box<VariationalSolver<RK45, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        case Integrator::BDF:
            return pbox::make_box<VariationalSolver<BDF, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        case Integrator::Euler:
            return pbox::make_box<VariationalSolver<Euler, T, N, SolverPolicy::RichVirtual, OdeType>>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        default:
            throw std::runtime_error("Unsupported integrator type");
    }
}

template<typename T, size_t N, SolverPolicy SP>
class ChaoticSolver : public std::conditional_t<traits::is_rich<SP>, OdeRichSolver<T, 2*N>, OdeSolver<T, 2*N>>{

public:

    virtual void    RhsMain(T* out, const T& t, const T* q) const = 0;

    virtual void    JacMain(T* out, const T& t, const T* q, const T* dt = nullptr) const = 0;

    virtual T       elapsed_time() const = 0;

    virtual T       kick() const = 0;

    virtual T       period() const = 0;

    virtual T       log_ksi() const = 0;

    virtual T       lyapunov_exponent() const = 0;

    virtual T       stretching_number() const = 0;

};


template<typename T, size_t N, hasRhsFunc<T> OdeType>
struct VariationalOdeSys{

public:

    using DualType      = autodiff::AutoDiff<T, 1, N>;
    using VarDualType   = autodiff::AutoDiff<T, 2, N>;
    

    // same as in BaseSolver, but we need to redefine it here for the variational system
    static constexpr JacPolicy MAIN_JP = (getJacPolicy<T, OdeType>() == JacPolicy::Autodiff && N == 0) ? JacPolicy::Exact : getJacPolicy<T, OdeType>();
    static constexpr JacPolicy JP = (MAIN_JP == JacPolicy::Autodiff) ? JacPolicy::Autodiff : JacPolicy::Approx;
    static_assert(MAIN_JP!=JacPolicy::Approx,  "VariationalSolver requires the base solver to have an exact jacobian for the original system");


    VariationalOdeSys(OdeType ode, size_t nsys, size_t nargs) : ode_(std::move(ode)), diff_worker(2*nsys), jac_worker(2*nsys), jm(nsys, nsys), diff_args(nargs), autodiff2_args(nargs), nsys(nsys) {
        if constexpr (N > 0){
            assert(N==nsys && "Incorrect number of equations in VariationalOdeSys");
        }

    }

    void    Rhs(T* out, const T& t, const T* q, const T* args) const{
        const T* delta_q = q + nsys;

        if constexpr (JP == JacPolicy::Autodiff){
            DualType* rhs = diff_worker.data();
            DualType* y = diff_worker.data() + N;
            FOR_LOOP(size_t, I, N,
                y[I] = DualType(q[I], autodiff::Variable<I>{});
            );

            for (size_t i=0; i<diff_args.size(); i++){
                diff_args[i] = DualType(args[i]);
            }

            ode_.Rhs(rhs, DualType(t), y, diff_args.data());

            std::fill(out+nsys, out+2*nsys, 0);
            FOR_LOOP(size_t, J, N,
                out[J] = rhs[J].value();
                FOR_LOOP(size_t, I, N,
                    out[I+N] += rhs[I].diff_value(J) * delta_q[J];
                );
            );
        } else {
            ode_.Rhs(out, t, q, args); //fills the first half (nsys) entries
            // fills jm with the jacobian of the original system at (t, q)
            // this should not call Base::jac_approx since we have demanded that the base solver has an exact jacobian for the original system
            ode_.Jac(jm.data(), t, q, args);
            for (size_t i=0; i<nsys; i++){
                out[i+nsys] = 0;
                for (size_t j=0; j<nsys; j++){
                    out[i+nsys] += jm(i, j) * q[nsys+j];
                }
            }
        }
    }

    // Only provided if it does not require finite differences, otherwise the base solver with automatically use jac_approx to compute the jacobian of the full system.
    void    Jac(T* out, const T& t, const T* q, const T* args, const T* dt = nullptr) const requires (JP == JacPolicy::Autodiff) {

        // If autodiff is enabled, then N is positive, so the system size is known at compile time. BaseSolver only allows autodiff if N > 0, so we can safely use a fixed-size array for the autodiff worker.
        VarDualType* rhs = jac_worker.data();
        VarDualType* y = jac_worker.data() + N;

        // copy the input state vector to the worker
        for (size_t i=0; i<N; i++){
            y[i] = VarDualType(q[i]);
        }

        // copy args
        for (size_t i=0; i<autodiff2_args.size(); i++){
            autodiff2_args[i] = VarDualType(args[i]);
        }

        // compute the jacobian using autodiff
        ode_.template Rhs<VarDualType>(rhs, VarDualType(t), y, autodiff2_args.data());

        // extract the jacobian matrix from the autodiff output
        ndspan::MutView<T, ndspan::Layout::F, 2*N, 2*N> m(out);
        FOR_LOOP(size_t, I, N,
            FOR_LOOP(size_t, J, N,
                m(I, J) = m(I+N, J+N) = rhs[I].diff_value(J);
                m(I, J+N) = 0;
                //the bottom left block now
                T sum = 0;
                for (size_t K=0; K<N; K++){
                    sum += rhs[I].diff_value(K, J) * q[N+K];
                }
                m(I+N, J) = sum;
            );
        );
    }

    const OdeType& ode() const{
        return ode_;
    }

private:
    OdeType ode_;
    mutable Array1D<DualType, 2*N> diff_worker;
    mutable Array1D<VarDualType, 2*N> jac_worker;
    mutable Array2D<T, N, N, ndspan::Allocation::Auto, ndspan::Layout::F> jm;
    mutable Array1D<DualType> diff_args;
    mutable Array1D<VarDualType> autodiff2_args;
    size_t nsys = N; // Size of the original system, without the variational equations (not the augmented system)
};


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
class VariationalSolver : public Solver<T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>> {


    /**
    This solvers integrates the coupled / augmented ODE system consisting of the original system and the variational equations. The state vector is of size 2*N, where the first N entries correspond to the state of the original system and the last N entries correspond to the deviation vector. The deviation vector is renormalized every "period" time units, and log_ksi keeps track of the logarithm of the stretching factor since the last renormalization. The user can retrieve the current value of the largest Lyapunov exponent using lyapunov_exponent(), which is computed as log_ksi divided by the elapsed time since the beginning of the integration.

    The augmented system is:

    dq/dt = f(q, t)
    d(delta_q)/dt = J(q, t) * delta_q

    The Jacobian of the original system is required to construct the variational equations.
    For implicit solvers, the Jacobian of the full system (original + variational) is also required.
    If the solver is provided with a templated Rhs function, then all jacobians are computed using autodiff. Otherwise, the solver must provide an exact jacobian for the original system, and finite differences are used to compute the jacobian of the full system, which schematically looks like:

    [           J           0  ]
    [ d(J)/dq *delta_q      J  ]

    */

public:

    using Base = Solver<T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>>;

    template<typename... Args>
    VariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args = {}, Args&&... extra);

    T elapsed_time() const{
        return this->t() - this->ics_ptr()[0];
    }

    T stretching_number() const{
        const size_t nsys = this->Nsys()/2;
        return log(norm(this->true_state_ptr()+2+nsys, nsys));
    }

    T kick() const{
        return stretching_number()/(this->t() - t_last_);
    }

    T period() const{
        return period_;
    }

    T log_ksi() const{
        return logksi_ + this->stretching_number();
    }

    T lyapunov_exponent() const{
        return np == 0 ? 0 : log_ksi()/elapsed_time();
    }

    void    Reset(){
        Base::Reset();
        ndspan::copy_array(tmp_state_.data(), this->ics().vector(), this->Nsys());
        t_last_ = this->ics_ptr()[0];
        t_next_ = t_last_ + period_*this->direction();
        np = 0;
        flagged = false;
        logksi_ =  0;
        logksi_last_ = 0;
    }

    void    ReAdjust(const T* new_vector){
        assert(false && "ReAdjust is not supported in VariationalSolver because it would interfere with the renormalization process. If you need to re-adjust the state at intermediate times, consider using a different solver or implementing a custom solution.");
    }

    void    RhsMain(T* out, const T& t, const T* q) const{
        this->ode().ode().Rhs(out, t, q, this->args().data()); //fills the first half (nsys) entries
    }

    void    JacMain(T* out, const T& t, const T* q, const T* dt = nullptr) const{
        if constexpr (hasJacFunc<OdeType, T>){
            this->ode().ode().Jac(out, t, q, this->args().data());
            return;
        } else {
            jac_approx<T>([this](T* out, const T& t, const T* q){
                this->RhsMain(out, t, q);
            }, out, worker.data(), t, q, dt, this->atol(), this->Nsys()/2);
        }
    }
    
protected:

    friend Base::MainSolverType;

    template<typename... Args>
    bool adv_impl(Args&&... args) {
        if (flagged){
            Base::ReAdjust(tmp_state_.data());
            flagged = false;
        }

        const int d = this->direction();
        const bool success = Base::adv_impl(t_next_, std::forward<Args>(args)...);
        if (success && (this->t() == t_next_)){
            const size_t nsys = this->Nsys()/2;
            t_last_ = t_next_;
            t_next_ = this->ics_ptr()[0] + (++np + 1)*period_*d;
            ndspan::copy_array(tmp_state_.data(), THIS->true_state_ptr()+2, 2*nsys);
            logksi_last_ = logksi_;
            logksi_ += log(norm(tmp_state_.data()+nsys, nsys));
            detail::normalized(tmp_state_.data(), tmp_state_.data(), nsys);
            flagged = true;
            return true;
        }else if (success){
            return true;
        }else{
            return false;
        }
    }

private:
    mutable Array1D<T, 4*N> worker;
    Array1D<T, 2*N> tmp_state_;
    T period_;
    T t_next_; // the next time at which to renormalize
    T t_last_;
    T logksi_ = 0; // log of the norm of the deviation vector at the last renormalization, used for computing the Lyapunov exponent
    T logksi_last_ = 0; // log of the norm of the deviation vector at the previous renormalization, used for computing the Lyapunov exponent
    size_t np = 0;
    bool flagged = false;
};





template<typename T, size_t N>
class VariationalODE : public ODE<T, N>{

public:

    using Base = ODE<T, N>;

    template<hasRhsFunc<T> OdeType>
    VariationalODE(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args = {}, std::vector<const Event<T>*> events = {}, Integrator method = Integrator::RK45) : Base(2*nsys){
        // Must create solver BEFORE register_state(), since it accesses solver_
        this->solver_ = get_virtual_variational_solver<T, N>(method, period, ode, t0, q0, delta_q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events);

        this->cached_idx_.resize(events.size(), 0);
        Base::register_state();
        for (size_t i=0; i<events.size(); i++){
            this->event_data_.allocate_event(events[i]->name());
        }
    }

    ODE<T, N>* clone() const override{
        return new VariationalODE<T, N>(*this);
    }

    const std::vector<T>& renorm_times() const{
        return renorm_times_;
    }

    const std::vector<T>& lyap_values() const{
        return lyap_values_;
    }

    const std::vector<T>& kick_values() const{
        return kick_values_;
    }

    DEFAULT_RULE_OF_FOUR(VariationalODE);

    void clear() override{
        Base::clear();
        renorm_times_ = std::vector<T>{};
        lyap_values_ = std::vector<T>{};
        kick_values_ = std::vector<T>{};
    }

    void reset() override{
        Base::reset();
        renorm_times_ = std::vector<T>{};
        lyap_values_ = std::vector<T>{};
        kick_values_ = std::vector<T>{};
    }

    const ChaoticSolver<T, N, SolverPolicy::RichVirtual>* solver() const {
        return static_cast<const ChaoticSolver<T, N, SolverPolicy::RichVirtual>*>(Base::solver());
    }

protected:

    void register_state() override{
        Base::register_state();
        renorm_times_.push_back(this->solver()->t());
        lyap_values_.push_back(this->solver()->lyapunov_exponent());
        kick_values_.push_back(this->solver()->kick());
    }

private:

    std::vector<T> renorm_times_ = {};
    std::vector<T> lyap_values_ = {};
    std::vector<T> kick_values_ = {};

};




template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getVariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args);
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getVariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}, EVENTS events = {}){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
}

} // namespace ode::chaos


namespace ode{

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct traits::SolverVirtualTypeTraits<chaos::VariationalSolver<Solver, T, N, SP, OdeType, DerivedVS>, T, NBase, SP> {
    using type = std::conditional_t<SP == SolverPolicy::Virtual || SP == SolverPolicy::RichVirtual, chaos::ChaoticSolver<T, N, SP>, EmptySolver>;
};

} // namespace ode

#endif
