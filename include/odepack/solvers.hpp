#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include "dop853.hpp"
#include "bdf.hpp"
#include "euler.hpp"

namespace ode {

#define SolverTemplate template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (is_rich<SP>)
Solver<T, N, SP, RhsType, JacType> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}, EVENTS events = {}) {
    return Solver<T, N, SP, RhsType, JacType>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args, events);
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
requires (!is_rich<SP>)
Solver<T, N, SP, RhsType, JacType> getSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}) {
    return Solver<T, N, SP, RhsType, JacType>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args);
}

// template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType, typename... Extra>
// Solver<T, N, SP, RhsType, JacType> getSolver(RhsType rhs, JacType jac, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}, Extra... extra){
//     return Solver<T, N, SP, RhsType, JacType>(OdeData<RhsType, JacType>{.rhs=rhs, .jacobian=jac, .obj=nullptr}, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args, extra...);
// }

// template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType, typename... Extra>
// Solver<T, N, SP, RhsType, JacType> getSolver(RhsType rhs, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}, Extra... extra){
//     return Solver<T, N, SP, RhsType, JacType>(OdeData<RhsType, JacType>{.rhs=rhs, .jacobian=nullptr, .obj=nullptr}, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args, extra...);
// }

template<typename T, size_t N, typename RhsType, typename JacType>
std::unique_ptr<OdeRichSolver<T, N>> get_virtual_solver(const std::string& name, MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) {
    if (name == "Euler"){
        return std::make_unique<Euler<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ode, t0, q0, nsys, first_step, dir, args, events);
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
        return std::make_unique<BDF<T, N, SolverPolicy::RichVirtual, RhsType, JacType>>(ARGS, events);;
    }
    else{
        throw std::runtime_error("Unknown solver name: " + name);
    }
}

} // namespace ode

#endif