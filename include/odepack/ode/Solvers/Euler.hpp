#ifndef EULER_HPP
#define EULER_HPP

#include "../Core/RichBase.hpp"

namespace ode{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class Euler : public BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

public:

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeType ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}) requires (!is_rich<SP>);

    Euler(OdeType ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}, EVENTS events = {}) requires (is_rich<SP>);

    auto local_interp() const;

protected:

    // Constructor signature that follows the main constructor patter.
    Euler(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    Euler(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

private:

    using Base = BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;
    friend Base::MainSolverType;

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

    static constexpr Integrator integrator = Integrator::Euler;
    static constexpr bool IS_IMPLICIT = false;
    static constexpr int ERR_EST_ORDER = 1;
};


} // namespace ode

#endif