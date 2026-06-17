#ifndef PYCHAOS_IMPL_HPP
#define PYCHAOS_IMPL_HPP


#include "../lib/PyChaos.hpp"
#include "../../ode/Chaos/VariationalSolvers_impl.hpp"


namespace ode::python {


template<typename T>
ode::chaos::VariationalODE<T, 0>& PyVarODE::varode(){
    return *static_cast<ode::chaos::VariationalODE<T, 0>*>(this->ode);
}

template<typename T>
const ode::chaos::VariationalODE<T, 0>& PyVarODE::varode() const {
    return *static_cast<const ode::chaos::VariationalODE<T, 0>*>(this->ode);
}


template<typename T>
ode::chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* PyVarSolver::cast(){
    return static_cast<ode::chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>*>(this->s);
}

template<typename T>
const ode::chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* PyVarSolver::cast() const {
    return static_cast<const ode::chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>*>(this->s);
}

} // namespace ode::python

#endif // PYCHAOS_IMPL_HPP