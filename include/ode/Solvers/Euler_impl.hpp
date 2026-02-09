#ifndef EULER_IMPL_HPP
#define EULER_IMPL_HPP

#include "Euler.hpp"

namespace ode{


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
Euler<T, N, SP, RhsType, JacType>::Euler(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir, const std::vector<T>& args) requires (!is_rich<SP>) : Base(ode, t0, q0, nsys, 0, 0, 0, inf<T>(), stepsize, dir, args) {}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
Euler<T, N, SP, RhsType, JacType>::Euler(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir, const std::vector<T>& args, EVENTS events) requires (is_rich<SP>) : Base(ode, t0, q0, nsys, 0, 0, 0, inf<T>(), stepsize, dir, args, events) {}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void Euler<T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    return lin_interp(result, t, this->t_old(), this->t_new(), this->old_state_ptr()+2, this->new_state_ptr()+2, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void Euler<T, N, SP, RhsType, JacType>::adapt_impl(T* res){
    State<T> old = this->new_state();
    T& t_new = res[0];
    T& habs = res[1];
    T* vec = res+2;

    const T& stepsize = old.habs();
    t_new = old.t() + stepsize;
    habs = stepsize;

    this->rhs(vec, t_new, old.vector());
    for (size_t i=0; i<this->Nsys(); i++){
        // y2 = y1 + f*dt, f = dy/dt
        vec[i] = old.vector()[i] + vec[i]*stepsize;
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 std::unique_ptr<Interpolator<T, N>> Euler<T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    return std::make_unique<LocalInterpolator<T, N>>(this->t_old(), this->t_new(), this->old_state_ptr()+2, this->new_state_ptr()+2, this->Nsys(), bdr1, bdr2);
}

} // namespace ode

#endif // EULER_IMPL_HPP