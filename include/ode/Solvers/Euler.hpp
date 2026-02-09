#ifndef EULER_HPP
#define EULER_HPP

#include "../Core/RichBase.hpp"

namespace ode{

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
class Euler : public BaseDispatcher<Euler<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>{

public:

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}) requires (!is_rich<SP>);

    Euler(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}, EVENTS events = {}) requires (is_rich<SP>);

     std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

private:

    using Base = BaseDispatcher<Euler<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>;
    friend Base::MainSolverType;

    void                                        adapt_impl(T* res);

     void                                 interp_impl(T* result, const T& t) const;

    static constexpr const char* name = "Euler";
    static constexpr bool IS_IMPLICIT = false;
    static constexpr int ERR_EST_ORDER = 1;
};


} // namespace ode

#endif