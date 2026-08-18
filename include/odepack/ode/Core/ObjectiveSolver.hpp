#ifndef OBJECTIVE_SOLVER_HPP
#define OBJECTIVE_SOLVER_HPP

#include "SolverBase.hpp"
#include "odepack/ode/Tools.hpp"

namespace ode{

template<typename T, isObjFun<T> Callable>
struct ObjFunData{

    Callable func;
    T ftol = 0; // tolerance for root finding (0 means machine precision)
    int dir = 0; // 0 means any direction, 1 means increasing, -1 means decreasing

};

/*
ObjectiveSolver passes itself as the Derived type to the base solver class,
and as a result it is the most derived class in the CRTP hierarchy.
That means that any class deriving from ObjectiveSolver that overrides functions like `Adv_Impl` will not have those overrides called by the base solver class,.
For virtual inheritance this might not be the case, so it is best not to override any of the base solver functions, but only extend the functionality.
*/
template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
class ObjectiveSolver : public Solver<T, N, SP, OdeType, ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>>{

    using Base = Solver<T, N, SP, OdeType, ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>>;

public:

    static constexpr size_t NOBJ = sizeof...(ObjFun);

    template<typename... Args>
    ObjectiveSolver(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args);

    void Reset();

    bool is_at_objective() const;

    int current_objective() const;

protected:

    template<typename... Args>
    bool Adv_Impl(Args&&... args);

    void ReAdjust(const T* new_vector);

    bool RequestTimeFloor(T& out);

private:

    bool get_nearest_floor(T& out, size_t& idx) const;

    void cache_current_signs();

    std::array<T, NOBJ> values = {};
    std::array<int, NOBJ> cached_sign = {};
    std::array<bool, NOBJ> detected = {};
    Array1D<T, N> worker;
    std::tuple<ObjFunData<T, ObjFun>...> obj;
    int current_idx = -1;
};


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T> ObjFun>
class SingleObjectiveSolver : public ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun>{

    using Base = ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun>;
public:

    template<typename... Args>
    SingleObjectiveSolver(ObjFunData<T, ObjFun> data, OdeType ode, Args&&... args);

    template<typename... Args>
    SingleObjectiveSolver(ObjFun obj_fun, OdeType ode, Args&&... args);

};

template<SolverTemplate typename Solver, typename T, size_t N, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun, typename... Args>
auto getObjectiveSolver(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args);




}; // namespace ode

#endif // OBJECTIVE_SOLVER_HPP
