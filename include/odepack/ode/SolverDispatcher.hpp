#ifndef SOLVER_DISPATCHER_HPP
#define SOLVER_DISPATCHER_HPP

#include "Solvers/Solvers.hpp" // IWYU pragma: keep
#include "odepack/ode/Core/VirtualBase.hpp"

namespace ode {


template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={}, EventList<T> events = {});

template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={});

template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
BoxedSolver<T, N, UP> make_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args);

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
BoxedSolver<T, N, UtilPolicy::Virtual> make_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args);

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
BoxedSolver<T, N, UtilPolicy::RichVirtual> make_rich_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args);

} // namespace ode

#endif // SOLVER_DISPATCHER_HPP