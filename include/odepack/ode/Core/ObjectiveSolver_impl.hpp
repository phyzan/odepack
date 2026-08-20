#ifndef OBJECTIVE_SOLVER_IMPL_HPP
#define OBJECTIVE_SOLVER_IMPL_HPP

#include "ObjectiveSolver.hpp"

namespace ode{

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
template<typename... Args>
ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::ObjectiveSolver(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args) : Base(std::move(ode), std::forward<Args>(args)...), obj(std::move(funcs)){
    this->cache_current_signs();
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
void ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::Reset(){
    Base::Reset();
    cache_current_signs();
    detected.fill(false);
    current_idx = -1;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
bool ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::is_at_objective() const {
    return current_idx != -1;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
int ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::current_objective() const {
    return current_idx;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
template<typename... Args>
bool ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::Adv_Impl(Args&&... args){
    T nearest_floor;
    size_t idx;
    current_idx = -1;
    if (this->is_at_new_state()){
        bool success = Base::Adv_Impl(std::forward<Args>(args)...);
        if (success && this->get_nearest_floor(nearest_floor, idx)){
            if (this->t() == nearest_floor){
                detected[idx] = false;
                cached_sign[idx] = 0;
                current_idx = int(idx);
            }
            return true;
        } else {
            return success;
        }
    } else if (this->get_nearest_floor(nearest_floor, idx)){
        if (Base::Adv_Impl(nearest_floor, std::forward<Args>(args)...)){
            if (this->t() == nearest_floor){
                detected[idx] = false; //turn off for next step, since the goal was achieved
                cached_sign[idx] = 0;
                current_idx = int(idx);
            }
            return true;
        } else {
            return false;
        }
    } else {
        return Base::Adv_Impl(std::forward<Args>(args)...);
    }
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
void ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    cache_current_signs();
    detected.fill(false);
    current_idx = -1;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
bool ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::RequestTimeFloor(T& out){
    bool base_floor = Base::RequestTimeFloor(out);
    const int d = this->direction();
    T my_floor = this->t_new();
    NDSPAN_FOR_LOOP(I, NOBJ,
        const int old_sgn = cached_sign[I];
        const int new_sign = sgn(std::get<I>(obj).func(this->t_new(), this->vector().data()));
        cached_sign[I] = new_sign;
        detected[I] = false;
        if (old_sgn != 0){
            // means we were not in the objective function in the previous step
            const int dir = std::get<I>(obj).dir;
            bool crossed;
            if (dir==0){
                crossed = old_sgn*new_sign <= 0;
            } else if (dir*d == 1){
                crossed = old_sgn < 0 && new_sign >= 0;
            } else {
                crossed = old_sgn > 0 && new_sign <= 0;
            }
            if ((detected[I] = crossed)){
                values[I] = bisect<T, RootPolicy::Right>([&](const T& t){
                    this->interp_impl(worker.data(), t);
                    return std::get<I>(obj).func(t, worker.data());
                }, this->t_old(), this->t_new(), std::get<I>(obj).ftol);
                my_floor = this->minimum_time(my_floor, values[I]);
            }
        }
    );
    if (base_floor){
        out = this->minimum_time(my_floor, out);
    } else {
        out = my_floor;
    }
    return true;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
bool ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::get_nearest_floor(T& out, size_t& idx) const{
    bool found = false;
    NDSPAN_FOR_LOOP(I, NOBJ,
        if (detected[I]){
            if (!found){
                out = values[I];
                idx = I;
                found = true;
            } else if (values[I]*this->direction() < out*this->direction()){
                out = values[I];
                idx = I;
            }
        }
    );
    return found;
}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun>
void ObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun...>::cache_current_signs(){
    NDSPAN_FOR_LOOP(I, NOBJ,
        cached_sign[I] = sgn(std::get<I>(obj).func(this->t(), this->vector().data()));
    );
}


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T> ObjFun>
template<typename... Args>
SingleObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun>::SingleObjectiveSolver(ObjFunData<T, ObjFun> data, OdeType ode, Args&&... args) : Base(std::tuple{data}, std::move(ode), std::forward<Args>(args)...) {}

template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, isObjFun<T> ObjFun>
template<typename... Args>
SingleObjectiveSolver<Solver, T, N, SP, OdeType, ObjFun>::SingleObjectiveSolver(ObjFun obj_fun, OdeType ode, Args&&... args) : Base(std::tuple{ObjFunData{std::move(obj_fun), T{0.0}, 1}}, std::move(ode), std::forward<Args>(args)...) {}

template<SolverTemplate typename Solver, typename T, size_t N, hasRhsFunc<T> OdeType, isObjFun<T>... ObjFun, typename... Args>
auto getObjectiveSolver(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args){
    return ObjectiveSolver<Solver, T, N, SolverPolicy::Static, OdeType, ObjFun...>(std::move(funcs), std::move(ode), std::forward<Args>(args)...);
}

} // namespace ode

#endif // OBJECTIVE_SOLVER_IMPL_HPP
