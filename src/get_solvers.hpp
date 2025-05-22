#ifndef GET_SOLVERS_HPP
#define GET_SOLVERS_HPP

#include "rk_adaptive.hpp"

template<class T, int N>
OdeSolver<T, N>* getSolver(const SolverArgs<T, N>& S, const std::string& method){

    OdeSolver<T, N>* solver = nullptr;

    if (method == "RK23") {
        solver = new RK23<T, N>(S);
    }
    else if (method == "RK45") {
        solver = new RK45<T, N>(S);
    }
    else {
        throw std::runtime_error("Unknown solver method: "+method);
    }

    return solver;
}

#endif