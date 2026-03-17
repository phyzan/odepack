#ifndef SOLVER_DISPATCHER_IMPL_HPP
#define SOLVER_DISPATCHER_IMPL_HPP

#include "SolverDispatcher.hpp"

namespace ode{

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (is_rich<SP>)
Solver<T, N, SP, RhsType, JacType, void> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, EVENTS events) {
    return Solver<T, N, SP, RhsType, JacType, void>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events);
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (!is_rich<SP>)
Solver<T, N, SP, RhsType, JacType, void> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args) {
    return Solver<T, N, SP, RhsType, JacType, void>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args);
}


template<typename T, size_t N, typename RhsType, typename JacType>
std::unique_ptr<OdeRichSolver<T, N>> get_virtual_solver(Integrator method, MAIN_CONSTRUCTOR(T), EVENTS events) {

    switch (method){
        case Integrator::Euler:
            return std::make_unique<Euler<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ode, t0, q0, nsys, stepsize, dir, args, events);
        case Integrator::RK23:
            return std::make_unique<RK23<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
        case Integrator::RK45:
            return std::make_unique<RK45<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
        case Integrator::DOP853:
            return std::make_unique<DOP853<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
        case Integrator::BDF:
            return std::make_unique<BDF<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
        case Integrator::RK4:
            return std::make_unique<RK4<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events);
        default:
            throw std::runtime_error("Unknown integrator enum value");
    }
}


} // namespace ode

#endif // SOLVER_DISPATCHER_IMPL_HPP