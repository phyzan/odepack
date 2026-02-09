#ifndef VARIATIONAL_SOLVERS_IMPL_HPP
#define VARIATIONAL_SOLVERS_IMPL_HPP

#include "VariationalSolvers.hpp"

namespace ode{


template<typename T>
void normalized(T* q, size_t nsys){
    T N = norm(q+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        q[i+nsys] /= N;
    }
}

// NormalizationEvent implementations
template<typename T, typename Derived>
NormalizationEvent<T, Derived>::NormalizationEvent(const std::string& name, T period) : Base(name, period, nullptr, true, nullptr) {}

template<typename T, typename Derived>
const T& NormalizationEvent<T, Derived>::logksi() const{
    return _logksi;
}

template<typename T, typename Derived>
T NormalizationEvent<T, Derived>::lyap() const{
    return _logksi/(this->delta_t_abs());
}

template<typename T, typename Derived>
T NormalizationEvent<T, Derived>::delta_s() const{
    return _delta_s;
}

template<typename T, typename Derived>
size_t NormalizationEvent<T, Derived>::Nsys_main() const{
    return this->Nsys()/2;
}

template<typename T, typename Derived>
void NormalizationEvent<T, Derived>::register_impl(){
    //call Base::register_impl() first
    Base::register_impl();
    _delta_s = log(norm(this->state()->exposed().vector()+Nsys_main(), Nsys_main()));
    _logksi += _delta_s;
}

template<typename T, typename Derived>
void NormalizationEvent<T, Derived>::reset_impl(int direction){
    //call Base::reset_impl() first
    Base::reset_impl(direction);
    _logksi = 0;
    _delta_s = 0;
}

template<typename T, typename Derived>
void NormalizationEvent<T, Derived>::mask_impl(T* out, const T& t, const T* q) const{
    size_t nsys = this->Nsys_main();
    T N = norm(q+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        out[i] = q[i];
        out[i+nsys] = q[i+nsys]/N;
    }
}

template<typename T, typename Derived>
 bool NormalizationEvent<T, Derived>::is_masked_impl() const{
    return true;
}


template<typename T, size_t N, typename RhsType, typename JacType>
VariationalSolver<T, N, RhsType, JacType>::VariationalSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, const std::string& method) : Base() {
    Array1D<T, N, Allocation::Auto> tmp(q0_, 2*nsys);
    normalized<T>(tmp.data(), nsys);
    NormalizationEvent<T> event("Normalization", period);
    nsys *= 2;
    const T* q0 = tmp.data();
    this->_ptr = get_virtual_solver<T, N>(method, ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, {&event}).release();
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const NormalizationEvent<T>& VariationalSolver<T, N, RhsType, JacType>::main_event() const{
    return static_cast<const NormalizationEvent<T>&>(this->ptr->event_col().event(0));
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const T& VariationalSolver<T, N, RhsType, JacType>::logksi() const{
    return this->main_event().logksi();
}

template<typename T, size_t N, typename RhsType, typename JacType>
 T VariationalSolver<T, N, RhsType, JacType>::lyap() const{
    return this->main_event().lyap();
}

template<typename T, size_t N, typename RhsType, typename JacType>
 T VariationalSolver<T, N, RhsType, JacType>::t_lyap() const{
    return this->main_event().delta_t_abs();
}

template<typename T, size_t N, typename RhsType, typename JacType>
 T VariationalSolver<T, N, RhsType, JacType>::delta_s() const{
    return this->main_event().delta_s();
}

// VariationalODE implementations
template<typename T, size_t N, typename RhsType, typename JacType>
VariationalODE<T, N, RhsType, JacType>::VariationalODE(OdeData<RhsType, JacType> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, std::vector<const Event<T>*> events, const std::string& method) : ODE<T, N>(){
    for (size_t i=0; i<events.size(); i++){
        if (dynamic_cast<const NormalizationEvent<T>*>(events[i])){
            throw std::runtime_error("Initializing a VariationalOdeSolver requires that no normalization events are passed in the constructor");
        }
    }

    Array1D<T, N, Allocation::Auto> tmp(q0_, 2*nsys);
    normalized<T>(tmp.data(), nsys);
    NormalizationEvent<T> extra_event("Normalization", period);
    events.insert(events.begin(), &extra_event);
    nsys *= 2;
    const T* q0 = tmp.data();
    this->_init(ARGS, events, method);
}

template<typename T, size_t N, typename RhsType, typename JacType>
ODE<T, N>* VariationalODE<T, N, RhsType, JacType>::clone() const{
    return new VariationalODE<T, N, RhsType, JacType>(*this);
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const std::vector<T>& VariationalODE<T, N, RhsType, JacType>::t_lyap() const{
    return _t_lyap;
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const std::vector<T>& VariationalODE<T, N, RhsType, JacType>::lyap() const{
    return _lyap_array;
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const std::vector<T>& VariationalODE<T, N, RhsType, JacType>::kicks() const{
    return _delta_s_arr;
}

template<typename T, size_t N, typename RhsType, typename JacType>
void VariationalODE<T, N, RhsType, JacType>::clear(){
    ODE<T, N>::clear();
    _t_lyap.clear();
    _t_lyap.shrink_to_fit();
    _lyap_array.clear();
    _lyap_array.shrink_to_fit();
    _delta_s_arr.clear();
    _delta_s_arr.shrink_to_fit();
}

template<typename T, size_t N, typename RhsType, typename JacType>
void VariationalODE<T, N, RhsType, JacType>::reset(){
    ODE<T, N>::reset();
    _t_lyap.clear();
    _t_lyap.shrink_to_fit();
    _lyap_array.clear();
    _lyap_array.shrink_to_fit();
    _delta_s_arr.clear();
    _delta_s_arr.shrink_to_fit();
}

template<typename T, size_t N, typename RhsType, typename JacType>
 const NormalizationEvent<T>& VariationalODE<T, N, RhsType, JacType>::_main_event()const{
    return static_cast<const NormalizationEvent<T>&>(this->solver()->event_col().event(_ind));
}

template<typename T, size_t N, typename RhsType, typename JacType>
void VariationalODE<T, N, RhsType, JacType>::_register_event(size_t event){
    ODE<T, N>::_register_event(event);
    if (event == _ind){
        const NormalizationEvent<T>& ev = _main_event();
        _t_lyap.push_back(ev.delta_t_abs());
        _lyap_array.push_back(ev.lyap());
        _delta_s_arr.push_back(ev.delta_s());
    }
}

}

#endif