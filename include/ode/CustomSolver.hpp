#ifndef CUSTOM_SOLVER_HPP
#define CUSTOM_SOLVER_HPP


#include "SolverDispatcher.hpp"


namespace ode {

//e.g. <MyDerivedClass, RK45, T, N, SP, RhsType, JacType>
template<typename Derived, template<typename, size_t, SolverPolicy, typename, typename, typename> typename Solver, typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class CustomSolver : public Solver<T, N, SP, RhsType, JacType, Derived>{

protected:
    using Base = Solver<T, N, SP, RhsType, JacType, Derived>;

public:

    template<typename... Args>
    CustomSolver(Args&&... args) : Base(std::forward<Args>(args)...) {}

    inline void    Rhs(T* jm, const T& t, const T* q) const{
        Base::Rhs(jm, t, q);
    }

    inline void    Jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const{
        Base::Jac(jm, t, q, dt);
    }

};


} // namespace ode

#endif // CUSTOM_SOLVER_HPP