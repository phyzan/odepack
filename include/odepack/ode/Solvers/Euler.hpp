#ifndef EULER_HPP
#define EULER_HPP

#include "../Core/RichBase.hpp"

namespace ode{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class Euler : public detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = false;

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir=1, std::vector<T> args = {}) requires (!traits::is_rich<SP>);

    Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir=1, std::vector<T> args = {}, EventList<T> events = {}) requires (traits::is_rich<SP>);

    // Constructor signature that follows the main constructor pattern.
    Euler(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>);

    Euler(MAIN_DEFAULT_CONSTRUCTOR(T), EventList<T> events = {}) requires (traits::is_rich<SP>);

    Integrator method() const;

    auto local_interp() const;

protected:

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

};



template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::Euler, T, N, SP, OdeType, Derived>{
    using type = Euler<T, N, SP, OdeType, Derived>;
};


} // namespace ode

#endif