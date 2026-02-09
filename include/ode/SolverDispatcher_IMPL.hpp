#ifndef SOLVER_DISPATCHER_IMPL_HPP
#define SOLVER_DISPATCHER_IMPL_HPP

#include "SolverDispatcher.hpp"

namespace ode{

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (is_rich<SP>)
Solver<T, N, SP, RhsType, JacType> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, EVENTS events) {
    return Solver<T, N, SP, RhsType, JacType>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events);
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (!is_rich<SP>)
Solver<T, N, SP, RhsType, JacType> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args) {
    return Solver<T, N, SP, RhsType, JacType>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args);
}


template<typename T, size_t N, typename RhsType, typename JacType>
std::unique_ptr<OdeRichSolver<T, N>> get_virtual_solver(const std::string& name, MAIN_CONSTRUCTOR(T), EVENTS events) {
    if (name == "Euler"){
        return std::make_unique<Euler<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ode, t0, q0, nsys, stepsize, dir, args, events);
    }
    else if (name == "RK23") {
        return std::make_unique<RK23<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
    }
    else if (name == "RK45") {
        return std::make_unique<RK45<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
    }
    else if (name == "DOP853") {
        return std::make_unique<DOP853<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
    }
    else if (name == "BDF"){
        return std::make_unique<BDF<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);
    }
    else if (name == "RK4"){
        return std::make_unique<RK4<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events);
    }
    else{
        throw std::runtime_error("Unknown solver name: " + name);
    }
}


} // namespace ode

#endif // SOLVER_DISPATCHER_IMPL_HPP