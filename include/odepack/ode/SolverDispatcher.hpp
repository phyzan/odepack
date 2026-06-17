#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include "Solvers/Solvers.hpp"

namespace ode {


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}, EVENTS events = {});

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={});

template<typename T, size_t N, hasRhsFunc<T> OdeType>
std::unique_ptr<OdeRichSolver<T, N>> get_virtual_solver(Integrator method, MAIN_CONSTRUCTOR(T), EVENTS events);

} // namespace ode

#endif