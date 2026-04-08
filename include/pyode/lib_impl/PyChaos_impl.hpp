#ifndef PYCHAOS_IMPL_HPP
#define PYCHAOS_IMPL_HPP


#include "../lib/PyChaos.hpp"
#include "../../ode/Chaos/VariationalSolvers_impl.hpp"


namespace ode{


template<typename T>
VariationalODE<T, 0>& PyVarODE::varode(){
    return *static_cast<VariationalODE<T, 0>*>(this->ode);
}

template<typename T>
const VariationalODE<T, 0>& PyVarODE::varode() const {
    return *static_cast<const VariationalODE<T, 0>*>(this->ode);
}


template<typename T>
ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* PyVarSolver::cast(){
    return static_cast<ChaoticSolver<T, 0, SolverPolicy::RichVirtual>*>(this->s);
}

template<typename T>
const ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* PyVarSolver::cast() const {
    return static_cast<const ChaoticSolver<T, 0, SolverPolicy::RichVirtual>*>(this->s);
}

} // namespace ode

#endif // PYCHAOS_IMPL_HPP