#ifndef EULER_HPP
#define EULER_HPP

#include "../Core/RichBase.hpp"

namespace ode{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class Euler : public detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr Integrator INTEGRATOR = Integrator::Euler;
    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = false;

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeType ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}) requires (!traits::is_rich<SP>);

    Euler(OdeType ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}, EVENTS events = {}) requires (traits::is_rich<SP>);

    auto local_interp() const;

protected:

    // Constructor signature that follows the main constructor patter.
    Euler(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>);

    Euler(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (traits::is_rich<SP>);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

};


} // namespace ode

#endif