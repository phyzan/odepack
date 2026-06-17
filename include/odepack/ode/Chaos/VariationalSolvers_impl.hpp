#ifndef VARIATIONAL_SOLVERS_IMPL_HPP
#define VARIATIONAL_SOLVERS_IMPL_HPP

#include "VariationalSolvers.hpp"

namespace ode::chaos{



template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
VariationalSolver<Solver, T, N, SP, OdeType, Derived>::VariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, Args&&... extra) : Base(VariationalOdeSys<T, N, OdeType>(ode, nsys, args.size()), t0,
    !q0 || !delta_q0 ? nullptr : [&]() -> Array1D<T, 2*N> {
        Array1D<T, 2*N> tmp(2*nsys);
        ndspan::copy_array(tmp.data(), q0, nsys);
        ndspan::copy_array(tmp.data()+nsys, delta_q0, nsys);
        detail::normalized(tmp.data(), tmp.data(), nsys);
        return tmp;
    }().data(),
    2*nsys, rtol, atol, min_step, max_step, stepsize, dir, args, std::forward<Args>(extra)...), worker(4*nsys), tmp_state_(2*nsys), period_(period), t_next_(t0+period*dir), t_last_(t0) {

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
    ndspan::copy_array(tmp_state_.data(), this->ics().vector(), 2*nsys);

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

} // namespace detail

} // namespace ode

#endif