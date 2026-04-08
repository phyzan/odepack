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

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Type>
RK4<T, N, SP, OdeType, Derived>::RK4(MAIN_CONSTRUCTOR(T), Type&&... extras) : Base(ode, t0, q0, nsys, rtol, atol, 0, inf<T>(), stepsize, dir, args, std::forward<Type>(extras)...),
#ifdef RK4_DENSE
K(9, nsys)
#else
K(5, nsys)
#endif
{
    // min_step and max_step are not used in RK4
}


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK4<T, N, SP, OdeType, Derived>::local_interp() const{
    size_t nsys = this->Nsys();
#ifdef RK4_DENSE
    return [solver=*this](T* out, const T& t){
            rk4_step([&solver](T* out, const T& tt, const T* y) LAMBDA_INLINE{
            solver.rhs(out, tt, y);
        }, out, solver.t_old(), t - solver.t_old(), solver.old_state_ptr()+2, solver.K.data()+5*nsys, nsys, solver.K.data() + 4*nsys);
    };
#else
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    return [n=nsys, t1=this->t_old(), t2 = d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, nsys), y2=Array1D<T, N>(d+2, nsys), y1dot=Array1D<T, N>(K.data(), nsys), y2dot=Array1D<T, N>(K.data()+nsys, nsys)](T* out, const T& t){
        rk4_interp(out, t, t1, t2, y1.data(), y2.data(), y1dot.data(), y2dot.data(), n);
    };
#endif
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK4<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    // standard Runge-Kutta-4 with fixed step size

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
    if (!all_are_finite(y_new, n)){
        return StepResult::INF_ERROR;
    }else{
        return StepResult::Success;
    }

}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
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


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K.fill(0);
    interp_data_set = false;

}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    set_interp_data();
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::set_interp_data() const{
    if (!interp_data_set){
        const T* d = this->interp_new_state_ptr();
        this->rhs(K.data()+this->Nsys(), d[0], d+2);
        interp_data_set = true;
    }
}

} // namespace ode

#endif // RUNGEKUTTA_IMPL_HPP