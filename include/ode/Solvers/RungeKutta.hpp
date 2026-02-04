#ifndef RUNGEKUTTA_HPP
#define RUNGEKUTTA_HPP


#include "../Core/RichBase.hpp"

namespace ode{

template<typename T>
void rk4_interp(T* out, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n);


template<typename T, size_t N>
class RK4Interpolator final : public LocalInterpolator<T, N>{

    using Base = LocalInterpolator<T, N>;

public:

    RK4Interpolator() = delete;
    DEFAULT_RULE_OF_FOUR(RK4Interpolator);

    RK4Interpolator(const T& t, const T* q, size_t nsys) : Base(t, q, nsys) {}

    RK4Interpolator(const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n, int bdr1, int bdr2) : Base(t1, t2, y1, y2, n, bdr1, bdr2), y1dot(y1dot, n), y2dot(y2dot, n) {}

    size_t                  order() const final{ return 4;}

    RK4Interpolator<T, N>*  clone() const final{ return new RK4Interpolator(*this);}

private:

    INLINE void _call_impl(T* result, const T& t) const final;

    Array1D<T, N> y1dot;
    Array1D<T, N> y2dot;

};

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
class RK4 : public BaseDispatcher<RK4<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>{

public:

    RK4(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RK4(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    inline VirtualInterp<T, N>  state_interpolator(int bdr1, int bdr2) const;

private:

    using Base = BaseDispatcher<RK4<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>;
    friend typename Base::MainSolverType;

    static constexpr const char*    name = "RK4";
    static constexpr bool           IS_IMPLICIT = false;
    static constexpr int            ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;

    inline void                 adapt_impl(T* res);

    inline void                 interp_impl(T* result, const T& t) const;

    inline void                 reset_impl();

    inline void                 re_adjust_impl(const T* new_vector);

    void set_interp_data() const;

    mutable Array2D<T, 5, N>    K;  // 4 stages of size N, plus one auxiliary array
    mutable bool        interp_data_set = false;

};


//==========================================================
//==================== IMPLEMENTATIONS =====================
//==========================================================


template<typename T>
void rk4_interp(T* out, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n){
    T h = t2 - t1;
    T theta = (t - t1) / h;

    T x2 = theta * theta;
    T x3 = x2 * theta;
    T h00 = 1 - 3*x2 + 2*x3;
    T h10 = theta - 2*x2 + x3;
    T h01 = 3*x2 - 2*x3;
    T h11 = -x2 + x3;

    for (size_t i=0; i<n; i++){
        out[i] = h00 * y1[i] + h * h10 * y1dot[i] + h01 * y2[i] + h * h11 * y2dot[i];
    }
}


template<typename T, size_t N>
INLINE void RK4Interpolator<T, N>::_call_impl(T* result, const T& t) const{
    rk4_interp(result, t, this->_t_min(), this->_t_max(), this->q_start().data(), this->q_end().data(), this->y1dot.data(), this->y2dot.data(), this->array_size());
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK4<T, N, SP, RhsType, JacType>::RK4(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ode, t0, q0, nsys, rtol, atol, 0, inf<T>(), first_step, dir, args), K(5, nsys){
    // min_step and max_step are not used in RK4
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK4<T, N, SP, RhsType, JacType>::RK4(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ode, t0, q0, nsys, rtol, atol, 0, inf<T>(), first_step, dir, args, events), K(5, nsys){
    // min_step and max_step are not used in RK4
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline VirtualInterp<T, N> RK4<T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    size_t nsys = this->Nsys();
    return std::unique_ptr<Interpolator<T, N>>(new RK4Interpolator<T, N>(this->t_old(), d[0], this->old_state_ptr(), d+2, K.data(), K.data()+nsys, nsys, bdr1, bdr2));
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void RK4<T, N, SP, RhsType, JacType>::adapt_impl(T* res){
    // standard Runge-Kutta-4 with fixed step size
    const T* state = this->new_state_ptr();
    const T& t = state[0];
    T h = state[1] * this->direction();
    const T* y = state + 2;
    size_t n = this->Nsys();

    res[0] = t + h; // t_new
    T* y_new = res + 2;

    // ======= Perform RK4 core algorithm =======
    T* k1 = K.data();
    T* k2 = K.data() + n;
    T* k3 = K.data() + 2*n;
    T* k4 = K.data() + 3*n;
    T* y_aux = K.data() + 4*n;

    this->rhs(k1, t, y);
    for (size_t i=0; i<n; i++){
        y_aux[i] = y[i] + h * k1[i] / 2;
    }
    this->rhs(k2, t + h/2, y_aux);
    for (size_t i=0; i<n; i++){
        y_aux[i] = y[i] + h * k2[i] / 2;
    }
    this->rhs(k3, t + h/2, y_aux);
    for (size_t i=0; i<n; i++){
        y_aux[i] = y[i] + h * k3[i];
    }
    this->rhs(k4, t + h, y_aux);
    for (size_t i=0; i<n; i++){
        y_new[i] = y[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6;
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void RK4<T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    size_t nsys = this->Nsys();
    rk4_interp(result, t, this->t_old(), d[0], this->old_state_ptr(), d+2, K.data(), K.data()+nsys, nsys);
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void RK4<T, N, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    K.set(0);
    interp_data_set = false;

}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void RK4<T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
    Base::re_adjust_impl(new_vector);
    set_interp_data();
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RK4<T, N, SP, RhsType, JacType>::set_interp_data() const{
    if (!interp_data_set){
        const T* d = this->interp_new_state_ptr();
        this->rhs(K.data()+this->Nsys(), d[0], d+2);
        interp_data_set = true;
    }
}

} // namespace ode


#endif // RUNGEKUTTA_HPP
