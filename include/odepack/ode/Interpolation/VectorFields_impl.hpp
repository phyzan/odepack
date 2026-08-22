#ifndef VECTOR_FIELDS_IMPL_HPP
#define VECTOR_FIELDS_IMPL_HPP


#include "VectorFields.hpp"

namespace ode::interp {

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
bool VectorField<Derived, T, NDIM, AS_VIRTUAL>::interp(T* out, const T* coords) const{
    return THIS->interp(out, coords);
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
int VectorField<Derived, T, NDIM, AS_VIRTUAL>::ndim() const {
    return THIS->ndim();
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
bool VectorField<Derived, T, NDIM, AS_VIRTUAL>::contains(const T* coords) const{
    return THIS->contains(coords);
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::OdeFuncNorm(T* out, const T& /*t*/, const T* q) const{
    size_t nd = this->ndim();
    if (!this->interp(out, q)){
        std::fill(out, out + nd, 0);
        return;
    }
    T norm = 0;
    for (size_t i = 0; i < nd; i++) {
        norm += out[i] * out[i];
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < nd; i++) {
        out[i] /= norm;
    }
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::OdeFunc(T* out, const T& /*t*/, const T* q) const{
    size_t nd = this->ndim();
    if (!this->interp(out, q)){
        std::fill(out, out + nd, 0);
    }
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
pbox::Box<OdeResult<T>> VectorField<Derived, T, NDIM, AS_VIRTUAL>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized, const std::vector<double>& t_eval) const{
    pbox::Box<ODE<T, NDIM>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<T>> result = pbox::make_box<OdeResult<T>>();
    ode->integrate(result.get_raw_pointer(), length, t_eval);
    return result;
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
pbox::Box<OdeResult<T>> VectorField<Derived, T, NDIM, AS_VIRTUAL>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized) const{
    pbox::Box<ODE<T, NDIM>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<T>> result = pbox::make_box<OdeResult<T>>();
    ode->integrate(result.get_raw_pointer(), length);
    return result;
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
pbox::Box<ODE<T, NDIM>> VectorField<Derived, T, NDIM, AS_VIRTUAL>::get_streamline_ode(const T* x0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized) const{
    if (normalized){
        return pbox::make_box<ODE<T, NDIM>>(OdeData{.Rhs=[this](T* out, const T& t, const T* q){ THIS->OdeFuncNorm(out, t, q); }}, T{0}, View1D<T, NDIM>{x0, this->ndim()}, rtol, atol, min_step, max_step, stepsize, direction, EventList<T>{}, method);
    } else {
        return pbox::make_box<ODE<T, NDIM>>(OdeData{.Rhs=[this](T* out, const T& t, const T* q){ THIS->OdeFunc(out, t, q); }}, T{0}, View1D<T, NDIM>{x0, this->ndim()}, rtol, atol, min_step, max_step, stepsize, direction, EventList<T>{}, method);
    }
}

} // namespace ode::interp

#endif // VECTOR_FIELDS_IMPL_HPP