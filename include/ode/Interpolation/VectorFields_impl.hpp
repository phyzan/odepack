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
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::ode_func_norm(T* out, const T& t, const T* q, const T* args, const void* ptr){
    assert(ptr != nullptr && "pointer is null");
    const auto* self_derived = reinterpret_cast<const Derived*>(ptr);
    const VectorField<Derived, T, NDIM, AS_VIRTUAL>* self = self_derived; // casting to base to expose the base interface for assisted typing, not necessary
    size_t nd = self->ndim();
    if (!self->interp(out, q)){
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
void VectorField<Derived, T, NDIM, AS_VIRTUAL>::ode_func(T* out, const T& t, const T* q, const T* args, const void* ptr){
    assert(ptr != nullptr && "pointer is null");
        const auto* self = reinterpret_cast<const Derived*>(ptr);
    size_t nd = self->ndim();
    if (!self->interp(out, q)){
        std::fill(out, out + nd, 0);
    }
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
OdeResult<T> VectorField<Derived, T, NDIM, AS_VIRTUAL>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method, bool normalized) const{
    auto* ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    OdeResult<T> result = ode->integrate(length, t_eval);
    delete ode;
    return result;
}


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
ODE<T, NDIM>* VectorField<Derived, T, NDIM, AS_VIRTUAL>::get_streamline_ode(const T* x0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const std::string& method, bool normalized) const{
    auto func = normalized ? ode_func_norm : ode_func;
        return new ODE<T, NDIM>(OdeData<Func<T>, void>{.rhs=func, .obj=static_cast<const Derived*>(this)}, 0, x0, this->ndim(), rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);
}

} // namespace ode

#endif // VECTOR_FIELDS_IMPL_HPP