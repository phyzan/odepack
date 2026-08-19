#ifndef SOLVER_DISPATCHER_IMPL_HPP
#define SOLVER_DISPATCHER_IMPL_HPP

#include "SolverDispatcher.hpp"
#include "odepack/ode/Core/VirtualBase.hpp"
#include "odepack/ode/IntegratorEnum.hpp"

namespace ode{

template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
inline Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, std::vector<T> args, EventList<T> events) {
    return Solver<T, N, SP, OdeType, void>(std::move(ode), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args), std::move(events));
}

template<SolverTemplate typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
inline Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, std::vector<T> args) {
    return Solver<T, N, SP, OdeType, void>(std::move(ode), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args));
}


template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UP> make_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    constexpr SolverPolicy SP = UP == UtilPolicy::RichVirtual ? SolverPolicy::RichVirtual : SolverPolicy::Virtual;
    return choose_integrator_case<BoxedSolver<T, N, UP>>(method,
        [&]<Integrator M>(){
            using Solver = typename SolverTypeGetter<M, T, N, SP, OdeType>::type;
            return pbox::make_box<Solver>(std::move(ode), t0, q0, std::forward<Args>(args)...);
        }
    );
}

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UtilPolicy::Virtual> make_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    return make_solver<UtilPolicy::Virtual>(method, std::move(ode), t0, q0, std::forward<Args>(args)...);
}

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UtilPolicy::RichVirtual> make_rich_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    return make_solver<UtilPolicy::RichVirtual>(method, std::move(ode), t0, q0, std::forward<Args>(args)...);
}


} // namespace ode

#endif // SOLVER_DISPATCHER_IMPL_HPP