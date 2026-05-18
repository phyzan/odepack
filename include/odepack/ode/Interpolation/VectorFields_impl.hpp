#ifndef VECTOR_FIELDS_IMPL_HPP
#define VECTOR_FIELDS_IMPL_HPP


#include "VectorFields.hpp"

namespace ode {

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
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::OdeFuncNorm(T* out, const T& t, const T* q, const T* args) const{
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
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::OdeFunc(T* out, const T& t, const T* q, const T* args) const{
    size_t nd = this->ndim();
    if (!this->interp(out, q)){
        std::fill(out, out + nd, 0);
    }
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
OdeResult<T> VectorField<Derived, T, NDIM, AS_VIRTUAL>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized, const std::vector<double>& t_eval) const{
    auto* ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    OdeResult<T> result;
    ode->integrate(&result, length, t_eval);
    delete ode;
    return result;
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
OdeResult<T> VectorField<Derived, T, NDIM, AS_VIRTUAL>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized) const{
    auto* ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    OdeResult<T> result;
    ode->integrate(&result, length);
    delete ode;
    return result;
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
ODE<T, NDIM>* VectorField<Derived, T, NDIM, AS_VIRTUAL>::get_streamline_ode(const T* x0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized) const{
    if (normalized){
        return new ODE<T, NDIM>(OdeData{.Rhs=[this](T* out, const T& t, const T* q, const T* args){ THIS->OdeFuncNorm(out, t, q, args); }}, 0, x0, this->ndim(), rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);
    } else {
        return new ODE<T, NDIM>(OdeData{.Rhs=[this](T* out, const T& t, const T* q, const T* args){ THIS->OdeFunc(out, t, q, args); }}, 0, x0, this->ndim(), rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);
    }
}

} // namespace ode

#endif // VECTOR_FIELDS_IMPL_HPP