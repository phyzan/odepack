#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include "dop853.hpp"
#include "stiff.hpp"

template<typename T, size_t N>
std::unique_ptr<OdeSolver<T, N>> get_solver(const std::string& name, MAIN_DEFAULT_CONSTRUCTOR(T, N)) {

    if (name == "RK23") {
        return std::make_unique<RK23<T, N>>(ARGS);
    }
    else if (name == "RK45") {
        return std::make_unique<RK45<T, N>>(ARGS);
    }
    else if (name == "DOP853") {
        return std::make_unique<DOP853<T, N>>(ARGS);
    }
    else if (name == "BDF"){
        return std::make_unique<BDF<T, N>>(ARGS);;
    }
    else{
        throw std::runtime_error("Unknown solver name: " + name);
    }
}

#endif