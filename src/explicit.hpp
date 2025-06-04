#ifndef EXPLICIT_HPP
#define EXPLICIT_HPP


#include "odesolvers.hpp"

template<class T, int N, class StateDerived>
class ExplicitSolver : public DerivedSolver<T, N, StateDerived>{

    using STATE = StateDerived;

protected:

    ExplicitSolver(DERIVED_SOLVER_CONSTRUCTOR(T, N)) : DerivedSolver<T, N, StateDerived>(name, PARTIAL_ARGS, err_est_ord, initial_state) {}

};



#endif