#ifndef PYODE_IMPL_HPP
#define PYODE_IMPL_HPP

#include "../bindings/PyOde.hpp"

namespace ode{

template<typename T, typename RhsType, typename JacType>
PyODE::PyODE(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, const std::vector<const Event<T>*>& events, const std::string& method) : DtypeDispatcher(get_scalar_type<T>()){
    data.is_lowlevel = true;
    data.shape = {py::ssize_t(nsys)};
    this->ode = new ODE<T, 0>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events, method);
}

template<typename T>
ODE<T>* PyODE::cast(){
    return reinterpret_cast<ODE<T>*>(this->ode);
}

template<typename T>
const ODE<T>* PyODE::cast() const {
    return reinterpret_cast<const ODE<T>*>(this->ode);
}

}

#endif // PYODE_IMPL_HPP