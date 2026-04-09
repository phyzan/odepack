#ifndef VARIATIONAL_SOLVERS_IMPL_HPP
#define VARIATIONAL_SOLVERS_IMPL_HPP

#include "VariationalSolvers.hpp"

namespace ode{



template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
VariationalSolver<Solver, T, N, SP, OdeType, Derived>::VariationalSolver(OdeType ode, T t0, const T* q0, const T* delta_q0, size_t nsys, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, Args&&... extra) : Base(ode, t0,
    !q0 || !delta_q0 ? nullptr : [&]() -> Array1D<T, 2*N> {
        Array1D<T, 2*N> tmp(2*nsys);
        ndspan::copy_array(tmp.data(), q0, nsys);
        ndspan::copy_array(tmp.data()+nsys, delta_q0, nsys);
        normalized(tmp.data(), tmp.data(), nsys);
        return tmp;
    }().data(),
    2*nsys, rtol, atol, min_step, max_step, stepsize, dir, args, std::forward<Args>(extra)...), jm(nsys, nsys), tmp_state_(2*nsys), autodiff_args(JP == JacPolicy::Autodiff ? args.size() : 0), period_(period), t_next_(t0+period*dir), t_last_(t0) {
    if (period <= 0){
        throw std::runtime_error("The renormalization period must be positive");
    }

    if constexpr (is_rich<SP>){
        //make sure there are no masked events, as they would interfere with the renormalization times.
        for (size_t i=0; i<this->event_col().size(); i++){
            if (this->event_col().event(i).is_masked()){
                throw std::runtime_error("VariationalSolver does not support masked events, as they would interfere with the renormalization times.");
            }
        }
    }
    if constexpr (JP == JacPolicy::Autodiff){
        for (size_t i=0; i<args.size(); i++){
            autodiff_args[i] = VarDualType(args[i]);
        }
    }
    if constexpr (std::is_same_v<Derived, void>){
        // Create combined ics (base + variational)
        Array1D<T, 2*N> combined(2*nsys);
        ndspan::copy_array(combined.data(), q0, nsys);
        ndspan::copy_array(combined.data()+nsys, delta_q0, nsys);
        normalized(combined.data(), combined.data(), nsys);

        // Now validate with the COMBINED ics
        this->ValidateIt(t0, combined.data(), stepsize);
    }
}


template<typename T>
void normalized(T* out, const T* src, size_t nsys){
    T N = norm(src+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        out[i] = src[i];
        out[i+nsys] /= N;
    }
}

} // namespace ode

#endif