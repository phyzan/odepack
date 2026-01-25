#ifndef CUSTOMSOLVER_HPP
#define CUSTOMSOLVER_HPP

#include "solverbase.hpp"

namespace ode {

template<template<typename, size_t, SolverPolicy> typename Custom, template<typename, size_t, SolverPolicy, typename> typename Solver, typename T, size_t N, SolverPolicy SP>
class CustomSolver : public Solver<T, N, SP, Custom<T, N, SP>>{

    using Base = Solver<T, N, SP, Custom<T, N, SP>>;

public:

    CustomSolver(T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}) : Base(OdeData<T>{.rhs=nullptr, .jacobian=nullptr, .obj=nullptr}, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args){}

    inline void rhs_impl(T* dq_dt, const T& t, const T* q) const{
        static_assert(false, "rhs_impl must be overriden in a derived class");
    }

    inline void jac_impl(T* j, const T& t, const T* q, const T* dt) const{
        static_assert(!Base::IS_IMPLICIT, "jac_impl must be overriden when inheriting from an implicit solver (e.g. BDF)");
    }

    inline void set_obj(const void* obj){
        static_assert(false, "set_obj does not do anything in CustomSolver");
    }

};


/*
Example


template<typename T, size_t N, SolverPolicy SP>
class MySolver : public CustomSolver<MySolver, RK45, T, N, SP>{

    using Base = CustomSolver<MySolver, RK45, T, N, SP>;

public:

    using Base::Base; // to inherit the constructor

    inline void rhs_impl(T* dq_dt, const T& t, const T* q) const{
        ...
    }

    inline void jac_impl(T* j, const T& t, const T* q, const T* dt) const{
        // only override this if it is needed
    }

};

*/

} // namespace ode

#endif
