#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include "Solvers/Solvers.hpp" // IWYU pragma: keep
#include "odepack/ode/Core/VirtualBase.hpp"

namespace ode {


template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={}, EventList<T> events = {});

template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={});

template<typename T, size_t N, UtilPolicy UP, hasRhsFunc<T> OdeType, typename... Args>
BoxedSolver<T, N, UP> make_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args);

} // namespace ode

#endif