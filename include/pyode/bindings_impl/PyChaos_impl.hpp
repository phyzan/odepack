#ifndef PYCHAOS_IMPL_HPP
#define PYCHAOS_IMPL_HPP


#include "../bindings/PyChaos.hpp"

namespace ode{

template<typename T>
const NormalizationEvent<T>& PyVarSolver::main_event() const{
    return static_cast<const NormalizationEvent<T>&>(reinterpret_cast<const OdeRichSolver<T>*>(this->s)->event_col().event(0));
}


template<typename T>
VariationalODE<T, 0, Func<T>, Func<T>>& PyVarODE::varode(){
    return *static_cast<VariationalODE<T, 0, Func<T>, Func<T>>*>(this->ode);
}

template<typename T>
const VariationalODE<T, 0, Func<T>, Func<T>>& PyVarODE::varode() const {
    return *static_cast<const VariationalODE<T, 0, Func<T>, Func<T>>*>(this->ode);
}

} // namespace ode

#endif // PYCHAOS_IMPL_HPP