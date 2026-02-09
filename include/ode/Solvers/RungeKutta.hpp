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

#ifdef RK4_DENSE
template<typename T, size_t N, typename RhsType>
class RK4Interpolator final : public LocalInterpolator<T, N>{
#else
template<typename T, size_t N>
class RK4Interpolator final : public LocalInterpolator<T, N>{
#endif
    using Base = LocalInterpolator<T, N>;

public:

    RK4Interpolator() = delete;

    DEFAULT_RULE_OF_FOUR(RK4Interpolator);

#ifdef RK4_DENSE
    RK4Interpolator(RhsType rhs, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n, int bdr1, int bdr2) : Base(t1, t2, y1, y2, n, bdr1, bdr2), y1dot(y1dot, n), y2dot(y2dot, n), K(5*n), rhs(rhs) {}
#else
    RK4Interpolator(const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n, int bdr1, int bdr2) : Base(t1, t2, y1, y2, n, bdr1, bdr2), y1dot(y1dot, n), y2dot(y2dot, n){}
#endif
    size_t                  order() const final{ return 4;}

#ifdef RK4_DENSE
    RK4Interpolator<T, N, RhsType>*  clone() const final{ return new RK4Interpolator(*this);}
#else
    RK4Interpolator<T, N>*  clone() const final{ return new RK4Interpolator(*this);}
#endif

private:

     void _call_impl(T* result, const T& t) const final;

    Array1D<T, N> y1dot;
    Array1D<T, N> y2dot;
#ifdef RK4_DENSE
    mutable Array1D<T, 5*N> K;
    RhsType rhs;
#endif

};

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
class RK4 : public BaseDispatcher<RK4<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>{

public:

    template<typename... Type>
    RK4(MAIN_DEFAULT_CONSTRUCTOR(T), Type&&... extras);

     VirtualInterp<T, N>  state_interpolator(int bdr1, int bdr2) const;

private:

    using Base = BaseDispatcher<RK4<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>;
    friend typename Base::MainSolverType;

    static constexpr const char*    name = "RK4";
    static constexpr bool           IS_IMPLICIT = false;
    static constexpr int            ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;

     void                 adapt_impl(T* res);

     void                 interp_impl(T* result, const T& t) const;

     void                 reset_impl();

     void                 re_adjust_impl(const T* new_vector);

    void set_interp_data() const;

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
