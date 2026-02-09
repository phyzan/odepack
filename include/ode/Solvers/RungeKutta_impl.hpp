#ifndef RUNGEKUTTA_IMPL_HPP
#define RUNGEKUTTA_IMPL_HPP

#include "RungeKutta.hpp"

namespace ode{

template<typename T, typename RhsType>
 void rk4_step(RhsType&& rhs, T* y_new, const T& t, const T& h, const T* y, T* k, size_t n, T* worker){
    // rhs(out, t, y);

    // ======= Perform RK4 core algorithm =======
    T* k1 = k;
    T* k2 = k + n;
    T* k3 = k + 2*n;
    T* k4 = k + 3*n;

    rhs(k1, t, y);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k1[i] / 2;
    }
    rhs(k2, t + h/2, worker);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k2[i] / 2;
    }
    rhs(k3, t + h/2, worker);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k3[i];
    }
    rhs(k4, t + h, worker);
    for (size_t i=0; i<n; i++){
        y_new[i] = y[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6;
    }
}

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


#ifdef RK4_DENSE
template<typename T, size_t N, typename RhsType>
 void RK4Interpolator<T, N, RhsType>::_call_impl(T* result, const T& t) const{
    if (t == this->_t_min()){
        copy_array(result, this->q_start().data(), this->array_size());
        return;
    }else if (t == this->_t_max()){
        copy_array(result, this->q_end().data(), this->array_size());
        return;
    }
    rk4_step<T>(this->rhs, result, this->_t_min(), t - this->_t_min(), this->q_start().data(), K.data(), this->array_size(), K.data() + 4*this->array_size());
}
#else
template<typename T, size_t N>
 void RK4Interpolator<T, N>::_call_impl(T* result, const T& t) const{
    rk4_interp(result, t, this->_t_min(), this->_t_max(), this->q_start().data(), this->q_end().data(), this->y1dot.data(), this->y2dot.data(), this->array_size());
}
#endif


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
template<typename... Type>
RK4<T, N, SP, RhsType, JacType>::RK4(MAIN_CONSTRUCTOR(T), Type&&... extras) : Base(ode, t0, q0, nsys, rtol, atol, 0, inf<T>(), stepsize, dir, args, std::forward<Type>(extras)...),
#ifdef RK4_DENSE
K(9, nsys)
#else
K(5, nsys)
#endif
{
    // min_step and max_step are not used in RK4
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 VirtualInterp<T, N> RK4<T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    size_t nsys = this->Nsys();
#ifdef RK4_DENSE
    auto func = [this](T* out, const T& t, const T* y) LAMBDA_INLINE{
        this->rhs(out, t, y);
    };
    return std::unique_ptr<Interpolator<T, N>>(new RK4Interpolator<T, N, decltype(func)>(func, this->t_old(), d[0], this->old_state_ptr()+2, d+2, K.data(), K.data()+nsys, nsys, bdr1, bdr2));
#else
    return std::unique_ptr<Interpolator<T, N>>(new RK4Interpolator<T, N>(this->t_old(), d[0], this->old_state_ptr()+2, d+2, K.data(), K.data()+nsys, nsys, bdr1, bdr2));
#endif
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void RK4<T, N, SP, RhsType, JacType>::adapt_impl(T* res){
    // standard Runge-Kutta-4 with fixed step size

    const T* state = this->new_state_ptr();
    const T& t = state[0];
    T h = state[1] * this->direction();
    const T* y = state + 2;
    size_t n = this->Nsys();

    res[0] = t + h; // t_new
    T* y_new = res + 2;

    auto rhs_caller = [this](T* out, const T& t, const T* y) LAMBDA_INLINE{
        this->rhs(out, t, y);
    };

    rk4_step(rhs_caller, y_new, t, h, y, K.data(), n, K.data() + 4*n);

}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void RK4<T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    size_t nsys = this->Nsys();
#ifdef RK4_DENSE    
    rk4_step([this](T* out, const T& tt, const T* y) LAMBDA_INLINE{
        this->rhs(out, tt, y);
    }, result, this->t_old(), t - this->t_old(), this->old_state_ptr()+2, K.data()+5*nsys, nsys, K.data() + 4*nsys);
#else
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    rk4_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, K.data(), K.data()+nsys, nsys);
#endif
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void RK4<T, N, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    K.set(0);
    interp_data_set = false;

}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void RK4<T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
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

#endif // RUNGEKUTTA_IMPL_HPP