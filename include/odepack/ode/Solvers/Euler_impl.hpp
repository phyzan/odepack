#ifndef EULER_IMPL_HPP
#define EULER_IMPL_HPP

#include "Euler.hpp"

namespace ode{


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir, std::vector<T> args) requires (!traits::is_rich<SP>) : Base(ode, t0, q0, 0, 0, 0, inf<T>(), stepsize, dir, std::move(args)) {}


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir, std::vector<T> args, EventList<T> events) requires (traits::is_rich<SP>) : Base(ode, t0, q0, 0, 0, 0, inf<T>(), stepsize, dir, std::move(args), std::move(events)) {}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T /*rtol*/, T /*atol*/, T /*min_step*/, T /*max_step*/, T stepsize, int dir, std::vector<T> args) requires (!traits::is_rich<SP>) : Euler(ode, t0, q0, stepsize, dir, std::move(args)) {}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T /*rtol*/, T /*atol*/, T /*min_step*/, T /*max_step*/, T stepsize, int dir, std::vector<T> args, EventList<T> events) requires (traits::is_rich<SP>) : Euler(ode, t0, q0, stepsize, dir, std::move(args), std::move(events)) {}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Integrator Euler<T, N, SP, OdeType, Derived>::method() const{
    return Integrator::Euler;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void Euler<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    return lin_interp(result, t, this->t_old(), this->t_new(), this->old_state_ptr()+2, this->interp_new_state_ptr()+2, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult Euler<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    T& t_new = res[0];
    T& habs = res[1];
    T* vec = res+2;

    const T& stepsize = state[1];
    t_new = state[0] + stepsize;
    habs = stepsize;

    this->rhs(vec, t_new, state+2);
    for (size_t i=0; i<this->Nsys(); i++){
        // y2 = y1 + f*dt, f = dy/dt
        vec[i] = state[i+2] + vec[i]*stepsize;
    }
    if (!all_are_finite(vec, this->Nsys())){
        return StepResult::INF_ERROR;
    }else{
        return StepResult::Success;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto Euler<T, N, SP, OdeType, Derived>::local_interp() const{
    return [t1=this->t_old(), t2=this->t_new(), y1=Array1D<T, N>(this->old_state_ptr()+2, this->Nsys()), y2=Array1D<T, N>(this->interp_new_state_ptr()+2, this->Nsys()), n=this->Nsys()](T* out, const T& t){
        lin_interp(out, t, t1, t2, y1.data(), y2.data(), n);
    };
}


} // namespace ode

#endif // EULER_IMPL_HPP