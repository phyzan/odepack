#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP

#include "../Core/Events.hpp"
#include "../OdeInt.hpp"
#include "../Core/VirtualBase.hpp"

namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================


template<typename T, size_t N, SolverPolicy SP>
class ChaoticSolver : public std::conditional_t<is_rich<SP>, OdeRichSolver<T, 2*N>, OdeSolver<T, 2*N>>{

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





// out (size 2*nsys) and in (size 2*nsys) can be the same pointer
template<typename T>
void normalized(T* out, const T* src, size_t nsys);

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class VariationalSolver : public Solver<T, 2*N, SP, OdeType, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>> {


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

    using Base = Solver<T, 2*N, SP, OdeType, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>>;
    static_assert(Base::JP!=JacPolicy::Approx,  "VariationalSolver requires the base solver to have an exact jacobian for the original system");

    using VarDualType = autodiff::AutoDiff<T, 2, N>; //second order diffs required for the jacobian of the full system (enabled only if the base solver has a templated Rhs)

    // if Base has a templated rhs, then we can use autodiff to compute the jacobian of the full system (using 2nd order derivatives), otherwise we will have to use finite differences
    static constexpr JacPolicy JP = (Base::JP == JacPolicy::Autodiff) ? JacPolicy::Autodiff : JacPolicy::Approx;

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
        Base::Rhs(out, t, q); //fills the first half (nsys) entries
    }

    void    JacMain(T* out, const T& t, const T* q, const T* dt = nullptr) const{
        Base::Jac(out, t, q, dt);
    }

    // Derived classes that override this method must implement a Rhs function for the full system (the original system + the variational equations)
    void    Rhs(T* out, const T& t, const T* q) const{
        size_t nsys = this->Nsys()/2;
        Base::Rhs(out, t, q); //fills the first half (nsys) entries

        // fills jm with the jacobian of the original system at (t, q)
        // this should not call Base::jac_approx since we have demanded that the base solver has an exact jacobian for the original system
        Base::Jac(jm.data(), t, q);
        for (size_t i=0; i<nsys; i++){
            out[i+nsys] = 0;
            for (size_t j=0; j<nsys; j++){
                out[i+nsys] += jm(i, j) * q[nsys+j];
            }
        }
    }

    // Derived classes that override this method must implement a Jac function for the full system (the original system + the variational equations)
    // Important: "out" has a F-storage layout (column-major)
    void    Jac(T* out, const T& t, const T* q, const T* dt = nullptr) const{
        //we must call the base solver's jac_approx to fill the part of the jacobian corresponding to the original system, since Base::Jac only affects the original system
        if constexpr (JP == JacPolicy::Approx){
            Base::jac_approx(out, t, q, dt);
        } else {
            // If autodiff is enabled, then N is positive, so the system size is known at compile time. BaseSolver only allows autodiff if N > 0, so we can safely use a fixed-size array for the autodiff worker.
            VarDualType* rhs = autodiff_jac_worker.data();
            VarDualType* y = autodiff_jac_worker.data() + N;

            // copy the input state vector to the worker
            for (size_t i=0; i<2*this->Nsys(); i++){
                y[i] = VarDualType(q[i]);
            }

            // compute the jacobian using autodiff
            this->ode().template Rhs<VarDualType>(rhs, VarDualType(t), y, autodiff_args.data());

            // extract the jacobian matrix from the autodiff output
            ndspan::MutView<T, ndspan::Layout::F, N, N> m(out);
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
    }

protected:

    friend Base::MainSolverType;

    void    set_args_impl(const T* new_args){
        Base::set_args_impl(new_args);
        for (size_t i=0; i<this->args().size(); i++){
            autodiff_args[i] = VarDualType(new_args[i]);
        }
    }

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
            normalized(tmp_state_.data(), tmp_state_.data(), nsys);
            flagged = true;
            return true;
        }else if (success){
            return true;
        }else{
            return false;
        }
    }

    // q0 is the full vector
    bool ValidateIt(const T& t0, const T* q0, const T& stepsize) {
        // Initialize tmp_state_ from the now-valid ics
        if (Base::ValidateIt(t0, q0, stepsize)){
            ndspan::copy_array(tmp_state_.data(), this->ics().vector(), this->Nsys());
            return true;
        } else {
            return false;
        }
    }

private:
    
    mutable Array2D<T, N, N, ndspan::Allocation::Auto, ndspan::Layout::F> jm; // jacobian matrix for the original system
    Array1D<T, 2*N> tmp_state_;
    Array1D<VarDualType> autodiff_args;
    mutable Array1D<VarDualType, JP == JacPolicy::Autodiff ? 4*N : 0> autodiff_jac_worker;
    T period_;
    T t_next_; // the next time at which to renormalize
    T t_last_;
    T logksi_ = 0; // log of the norm of the deviation vector at the last renormalization, used for computing the Lyapunov exponent
    T logksi_last_ = 0; // log of the norm of the deviation vector at the previous renormalization, used for computing the Lyapunov exponent
    size_t np = 0;
    bool flagged = false;
};


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct SolverVirtualTypeTraits<VariationalSolver<Solver, T, N, SP, OdeType, DerivedVS>, T, NBase, SP> {
    using type = std::conditional_t<SP == SolverPolicy::Virtual || SP == SolverPolicy::RichVirtual, ChaoticSolver<T, N, SP>, EmptySolver>;
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
requires (!is_rich<SP>)
Solver<T, N, SP, OdeType, void> getVariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args);
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
requires (is_rich<SP>)
Solver<T, N, SP, OdeType, void> getVariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}, EVENTS events = {}){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(ode, t0, q0, delta_q0, nsys, period, rtol, atol, min_step, max_step, stepsize, dir, args, events);
}


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



} // namespace ode

#endif
