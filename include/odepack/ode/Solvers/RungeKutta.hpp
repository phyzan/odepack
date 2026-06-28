#ifndef RUNGEKUTTA_HPP
#define RUNGEKUTTA_HPP


#include "../Core/RichBase.hpp"

namespace ode{

// Compile with RK4_DENSE to use rk4 steps for interpolation instead of the default Hermite polynomials
// This increases accuracy at the cost of performance and memory

template<typename T, typename RhsType>
void rk4_step(RhsType&& rhs, T* y_new, const T& t, const T& h, const T* y, T* k, size_t n, T* worker);

template<typename T>
void rk4_interp(T* out, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n);


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK4 : public detail::BaseDispatcher<GetDerived<RK4<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK4<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    template<typename... Type>
    RK4(MAIN_DEFAULT_CONSTRUCTOR(T), Type&&... extras);

    auto  local_interp() const;

    void        Reset();

    static constexpr Integrator     INTEGRATOR = Integrator::RK4;
    static constexpr int            ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;
    static constexpr bool           IS_IMPLICIT = false;

protected:

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

    void        ReAdjust(const T* new_vector);

    void        set_interp_data() const;

    // 4 stages of size N, plus one auxiliary array. if RK4_DENSE, K has 4 extra stages for dense output. So visually K = [k1, k2, k3, k4, aux | k1, k2, l3, k4 ]
#ifdef RK4_DENSE
    mutable Array2D<T, 9, N>    K;
#else
    mutable Array2D<T, 5, N>    K;
#endif
    mutable bool        interp_data_set = false;

};



} // namespace ode


#endif // RUNGEKUTTA_HPP
