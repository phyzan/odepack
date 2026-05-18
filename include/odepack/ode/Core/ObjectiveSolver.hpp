#ifndef OBJECTIVE_SOLVER_HPP
#define OBJECTIVE_SOLVER_HPP

#include "SolverBase.hpp"

namespace ode{

template<typename T, isObjFun<T> Callable>
struct ObjFunData{
    
    Callable func;
    T ftol = 0; // tolerance for root finding (0 means machine precision)
    int dir = 0; // 0 means any direction, 1 means increasing, -1 means decreasing
    
};


template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived, isObjFun<T>... ObjFun>
class ObjectiveSolver : public Solver<T, N, SP, OdeType, GetDerived<ObjectiveSolver<Solver, T, N, SP, OdeType, Derived, ObjFun...>, Derived>>{

    using Base = Solver<T, N, SP, OdeType, GetDerived<ObjectiveSolver<Solver, T, N, SP, OdeType, Derived, ObjFun...>, Derived>>;

public:

    static constexpr size_t NOBJ = sizeof...(ObjFun);

    template<typename... Args>
    ObjectiveSolver(std::tuple<ObjFunData<T, ObjFun>...> funcs, Args&&... args) : Base(std::forward<Args>(args)...), obj(std::move(funcs)){}

    void Reset(){
        Base::Reset();
        cache_current_signs();
        detected.fill(false);
        current_idx = -1;
    }

    template<typename... Args>
    bool adv_impl(Args&&... args){
        T nearest_floor;
        size_t idx;
        current_idx = -1;
        if (this->is_at_new_state()){
            bool success = Base::adv_impl(std::forward<Args>(args)...);
            if (success && this->get_nearest_floor(nearest_floor, idx)){
                if (this->t() == nearest_floor){
                    detected[idx] = false;
                    cached_sign[idx] = 0;
                    current_idx = idx;
                }
                return true;
            } else {
                return success;
            }
        } else if (this->get_nearest_floor(nearest_floor, idx)){
            if (Base::adv_impl(nearest_floor, std::forward<Args>(args)...)){
                if (this->t() == nearest_floor){
                    detected[idx] = false; //turn off for next step, since the goal was achieved
                    cached_sign[idx] = 0;
                    current_idx = idx;
                }
                return true;
            } else {
                return false;
            }
        } else {
            return Base::adv_impl(std::forward<Args>(args)...);
        }
    }

    void ReAdjust(const T* new_vector){
        Base::ReAdjust(new_vector);
        cache_current_signs();
        detected.fill(false);
        current_idx = -1;
    }

    bool RequestTimeFloor(T& out){
        bool base_floor = Base::RequestTimeFloor(out);
        const int d = this->direction();
        T my_floor = this->t_new();
        FOR_LOOP(size_t, I, NOBJ,
            const int old_sgn = cached_sign[I];
            const int new_sign = sgn(std::get<I>(obj).func(this->t_new(), this->vector().data(), this->args().data()));
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
                        return std::get<I>(obj).func(t, worker.data(), this->args().data());
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

    bool is_at_objective() const {
        return current_idx != -1;
    }

private:

    bool get_nearest_floor(T& out, size_t& idx) const{
        bool found = false;
        FOR_LOOP(size_t, I, NOBJ,
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

    void cache_current_signs(){
        FOR_LOOP(size_t, I, NOBJ,
            cached_sign[I] = sgn(std::get<I>(obj).func(this->t(), this->vector().data(), this->args().data()));
        );
    }

    std::array<T, NOBJ> values;
    std::array<int, NOBJ> cached_sign = {};
    std::array<bool, NOBJ> detected = {};
    Array1D<T, N> worker;
    std::tuple<ObjFunData<T, ObjFun>...> obj;
    int current_idx = -1;
};

}; // namespace ode

#endif // OBJECTIVE_SOLVER_HPP