#ifndef VARIATIONAL_SOLVERS_IMPL_HPP
#define VARIATIONAL_SOLVERS_IMPL_HPP

#include "VariationalSolvers.hpp"
#include "odepack/ode/Tools.hpp"

namespace ode::chaos{

template<typename T, size_t N, hasRhsFunc<T> OdeType>
VariationalOdeSys<T, N, OdeType>::VariationalOdeSys(OdeType ode, size_t ode_nsys) : ode_(std::move(ode)), diff_worker(2*ode_nsys), jac_worker(2*ode_nsys), jm(ode_nsys, ode_nsys), nsys(ode_nsys) {
    if constexpr (N > 0){
        assert(N==ode_nsys && "Incorrect number of equations in VariationalOdeSys");
    }
}
template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Rhs(T* out, const T& t, const T* q) const{
    const T* delta_q = q + nsys;

    if constexpr (JP == JacPolicy::Autodiff){
        if constexpr (N > 0){
            DualType* rhs = diff_worker.data();
            DualType* y = diff_worker.data() + N;
            NDSPAN_FOR_LOOP(I, N,
                y[I] = DualType(q[I], {.axis=I});
            );

            ode_.Rhs(rhs, t, y);

            std::fill(out+nsys, out+2*nsys, 0);
            NDSPAN_FOR_LOOP(J, N,
                out[J] = rhs[J].value();
                NDSPAN_FOR_LOOP(I, N,
                    out[I+N] += rhs[I].get_diff_wrt(J) * delta_q[J];
                );
            );
        } else {
            const size_t nvars_default = DualType::get_default_nvars();
            DualType::set_default_nvars(nsys);
            DualType* rhs = diff_worker.data();
            DualType* y = diff_worker.data() + nsys;
            for (size_t i=0; i<nsys; i++){
                y[i] = DualType(q[i], {.axis=int(i)});
            }
            ode_.Rhs(rhs, t, y);
            std::fill(out+nsys, out+2*nsys, 0);
            for (size_t j=0; j<nsys; j++){
                out[j] = rhs[j].value();
                for (size_t i=0; i<nsys; i++){
                    out[i+nsys] += rhs[i].get_diff_wrt(j) * delta_q[j];
                }
            }
            DualType::set_default_nvars(nvars_default);
        }
    } else {
        ode_.Rhs(out, t, q); //fills the first half (nsys) entries
        // fills jm with the jacobian of the original system at (t, q)
        // this should not call Base::jac_approx since we have demanded that the base solver has an exact jacobian for the original system
        ode_.Jac(jm.data(), t, q);
        for (size_t i=0; i<nsys; i++){
            out[i+nsys] = 0;
            for (size_t j=0; j<nsys; j++){
                out[i+nsys] += jm(i, j) * q[nsys+j];
            }
        }
    }
}

// Only provided if it does not require finite differences, otherwise the base solver will automatically use jac_approx to compute the jacobian of the full system.
template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Jac(T* out, const T& t, const T* q, const T* dt) const requires (JP == JacPolicy::Autodiff) {

    assert(dt == nullptr && "VariationalSolver overrides Jacobian computation for templated r.h.s functions and uses autodiff, so passing the `dt` argument is not used and should be nullptr");

    if constexpr (N > 0){
        VarDualType* rhs = jac_worker.data();
        VarDualType* y = jac_worker.data() + N;

        // copy the input state vector to the worker
        for (size_t i=0; i<N; i++){
            y[i] = VarDualType(q[i], {.axis=i});
        }

        // compute the jacobian using autodiff
        ode_.Rhs(rhs, t, y);

        // extract the jacobian matrix from the autodiff output
        ndspan::MutView<T, ndspan::Layout::F, 2*N, 2*N> m(out);
        NDSPAN_FOR_LOOP(I, N,
            NDSPAN_FOR_LOOP(J, N,
                m(I, J) = m(I+N, J+N) = rhs[I].get_diff_wrt(J);
                m(I, J+N) = 0;
                //the bottom left block now
                T sum = 0;
                for (size_t K=0; K<N; K++){
                    sum += rhs[I].get_diff_wrt(K, J) * q[N+K];
                }
                m(I+N, J) = sum;
            );
        );
    } else {
        const size_t nvars_default = VarDualType::get_default_nvars();
        VarDualType::set_default_nvars(nsys);
        VarDualType* rhs = jac_worker.data();
        VarDualType* y = jac_worker.data() + nsys;

        for (size_t i=0; i<nsys; i++){
            y[i] = VarDualType(q[i], {.axis=i});
        }

        ode_.Rhs(rhs, t, y);

        ndspan::MutView<T, ndspan::Layout::F> m(out, 2*nsys, 2*nsys);
        for (size_t i=0; i<nsys; i++){
            for (size_t j=0; j<nsys; j++){
                m(i, j) = m(i+nsys, j+nsys) = rhs[i].get_diff_wrt(j);
                m(i, j+nsys) = 0;
                T sum = 0;
                for (size_t k=0; k<nsys; k++){
                    sum += rhs[i].get_diff_wrt(k, j) * q[nsys+k];
                }
                m(i+nsys, j) = sum;
            }
        }
        VarDualType::set_default_nvars(nvars_default);
    }
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
const OdeType& VariationalOdeSys<T, N, OdeType>::ode() const{
    return ode_;
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
VariationalSolver<Solver, T, N, SP, OdeType, Derived>::VariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, Args&&... extra) : Base(VariationalOdeSys<T, N, OdeType>(ode, q0.size()), t0,
    !q0.data() || !delta_q0.data() ?
    View1D<T, 2*N>{nullptr, 2*q0.size()} :
    View1D<T, 2*N>{
        join_arrays(q0, delta_q0).data(),
        2*q0.size()
    }, rtol, atol, min_step, max_step, stepsize, dir, std::forward<Args>(extra)...), worker(4*q0.size()), tmp_state_(2*q0.size()), period_(period), t_next_(t0+period*dir), t_last_(t0) {

    if (period <= 0){
        throw std::runtime_error("The renormalization period must be positive");
    }

    if constexpr (traits::is_rich<SP>){
        //make sure there are no masked events, as they would interfere with the renormalization times.
        for (size_t i=0; i<this->event_col().size(); i++){
            if (this->event_col().event(i).is_masked()){
                throw std::runtime_error("VariationalSolver does not support masked events, as they would interfere with the renormalization times.");
            }
        }
    }
    ndspan::copy_array(tmp_state_.data(), this->ics().vector(), 2*q0.size());

}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::elapsed_time() const{
    return this->t() - this->ics_ptr()[0];
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::stretching_number() const{
    const size_t nsys = this->nsys()/2;
    return log(norm(this->true_state_ptr()+2+nsys, nsys));
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::kick() const{
    return stretching_number()/(this->t() - t_last_);
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::period() const{
    return period_;
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::log_ksi() const{
    return logksi_ + this->stretching_number();
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<Solver, T, N, SP, OdeType, Derived>::lyapunov_exponent() const{
    return np == 0 ? T{0} : T(log_ksi()/elapsed_time());
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<Solver, T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    ndspan::copy_array(tmp_state_.data(), this->ics().vector(), this->nsys());
    t_last_ = this->ics_ptr()[0];
    t_next_ = t_last_ + period_*this->direction();
    np = 0;
    flagged = false;
    logksi_ =  0;
    logksi_last_ = 0;
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<Solver, T, N, SP, OdeType, Derived>::RhsMain(T* out, const T& t, const T* q) const{
    this->ode().ode().Rhs(out, t, q); //fills the first half (nsys) entries
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<Solver, T, N, SP, OdeType, Derived>::JacMain(T* out, const T& t, const T* q, const T* dt) const{
    if constexpr (hasJacFunc<OdeType, T>){
        this->ode().ode().Jac(out, t, q);
        return;
    } else {
        jac_approx<T>([this](T* jm, const T& t_dummy, const T* q_dummy){
            this->RhsMain(jm, t_dummy, q_dummy);
        }, out, worker.data(), t, q, dt, this->atol(), this->nsys()/2);
    }
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<Solver, T, N, SP, OdeType, Derived>::ReAdjust(const T* /*new_vector*/){
    assert(false && "ReAdjust is not supported in VariationalSolver because it would interfere with the renormalization process. If you need to re-adjust the state at intermediate times, consider using a different solver or implementing a custom solution.");
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
bool VariationalSolver<Solver, T, N, SP, OdeType, Derived>::Adv_Impl(Args&&... args) {
    if (flagged){
        Base::ReAdjust(tmp_state_.data());
        flagged = false;
    }

    const int d = this->direction();
    const bool success = Base::Adv_Impl(t_next_, std::forward<Args>(args)...);
    if (success && (this->t() == t_next_)){
        const size_t nsys = this->nsys()/2;
        t_last_ = t_next_;
        t_next_ = this->ics_ptr()[0] + (++np + 1UL)*period_*d;
        ndspan::copy_array(tmp_state_.data(), THIS->true_state_ptr()+2, 2*nsys);
        logksi_last_ = logksi_;
        logksi_ += log(norm(tmp_state_.data()+nsys, nsys));
        detail::normalized(tmp_state_.data(), tmp_state_.data(), nsys);
        flagged = true;
        return true;
    } else if (success){
        return true;
    } else {
        return false;
    }
}

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Array1D<T, 2*N> VariationalSolver<Solver, T, N, SP, OdeType, Derived>::join_arrays(View1D<T, N> q0, View1D<T, N> delta_q0){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size");
    Array1D<T, 2*N> tmp(2*q0.size());
    ndspan::copy_array(tmp.data(), q0.data(), q0.size());
    ndspan::copy_array(tmp.data()+q0.size(), delta_q0.data(), delta_q0.size());
    detail::normalized(tmp.data(), tmp.data(), q0.size());
    return tmp;
}


namespace detail{

template<typename T>
void normalized(T* out, const T* src, size_t nsys){
    T N = norm(src+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        out[i] = src[i];
        out[i+nsys] /= N;
    }
}

} // namespace ode::detail


template<typename T, size_t N>
template<hasRhsFunc<T> OdeType>
VariationalODE<T, N>::VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events, Integrator method) : Base(2*q0.size()){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size in VariationalODE");
    // Must create solver BEFORE register_state(), since it accesses solver_
    this->solver_ = make_variational_solver<UtilPolicy::RichVirtual>(method, ode, t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));

    const EventCollection<T>& event_coll = this->solver()->get_event_col();

    this->cached_idx_.resize(event_coll.size(), 0);
    Base::register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        this->event_data_.allocate_event(event_coll.event(i).name());
    }
}

template<typename T, size_t N>
std::unique_ptr<ODE<T, N>> VariationalODE<T, N>::clone() const{
    return std::make_unique<VariationalODE<T, N>>(*this);
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::renorm_times() const{
    return renorm_times_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::lyap_values() const{
    return lyap_values_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::kick_values() const{
    return kick_values_;
}

template<typename T, size_t N>
void VariationalODE<T, N>::clear(){
    Base::clear();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
void VariationalODE<T, N>::reset(){
    Base::reset();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* VariationalODE<T, N>::solver() const {
    return static_cast<const ChaoticSolver<T, N, UtilPolicy::RichVirtual>*>(Base::solver());
}

template<typename T, size_t N>
void VariationalODE<T, N>::register_state(){
    Base::register_state();
    renorm_times_.push_back(this->solver()->get_time());
    lyap_values_.push_back(this->solver()->get_lyapunov_exponent());
    kick_values_.push_back(this->solver()->get_kick());
}


template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args){

    constexpr SolverPolicy SP = UP == UtilPolicy::Virtual ? SolverPolicy::Virtual : SolverPolicy::RichVirtual;

    return choose_integrator_case<pbox::Box<ChaoticSolver<T, 2*N, UP>>>(method,
        [&]<Integrator M>(){
            using Solver = typename SolverTypeGetter<M, T, N, SP, OdeType>::type;
            return pbox::make_box<VariationalSolver<
                M,
                typename Solver::value_type,
                Solver::NSYS,
                SP,
                OdeType>>(std::move(ode), t0, q0, delta_q0, period, std::forward<Args>(args)...);
        }
    );
}


template<SolverPolicy SP, Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir);
}


template<SolverPolicy SP, Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events){
    return VariationalSolver<Solver, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));
}

} // namespace ode

#endif
