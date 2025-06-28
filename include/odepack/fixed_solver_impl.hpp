#ifndef FIXED_SOLVER_IMPL_HPP
#define FIXED_SOLVER_IMPL_HPP

#define FS_MAIN_DEFAULT_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T stepsize, const std::vector<T>& args={}, const std::vector<Event<T, N>*> events={}

#define FS_SOLVER_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T stepsize, const std::vector<T>& args, const std::vector<Event<T, N>*> events

#define FS_ODE_CONSTRUCTOR(T, N) FS_MAIN_DEFAULT_CONSTRUCTOR(T, N), std::string method="RK4"

#define FS_ARGS rhs, t0, q0, stepsize, args, events

//TODO: Add RK4 class


#endif