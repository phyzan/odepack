#ifndef EULER_HPP
#define EULER_HPP

#include "rich_solver.hpp"


// OdeData<T> ode, const T& t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}


template<typename T, size_t N, SolverPolicy SP, typename Derived=void>
class Euler : public BaseDispatcher<GetDerived<Euler<T, N, SP, Derived>, Derived>, T, N, SP>{

    using Base = BaseDispatcher<GetDerived<Euler<T, N, SP, Derived>, Derived>, T, N, SP>;

public:

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeData<T> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}) requires (!is_rich<SP>);

    Euler(OdeData<T> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir=1, const std::vector<T>& args = {}, EVENTS events = {}) requires (is_rich<SP>);

    inline std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

    void                                        adapt_impl(T* res);

    inline void                                 interp_impl(T* result, const T& t) const;

    static constexpr const char* name = "Euler";
    static constexpr bool IS_IMPLICIT = false;
    static constexpr int ERR_EST_ORDER = 1;
};

//==========================================================
//==================== IMPLMENTATIONS ======================
//==========================================================


template<typename T, size_t N, SolverPolicy SP, typename Derived>
Euler<T, N, SP, Derived>::Euler(OdeData<T> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir, const std::vector<T>& args) requires (!is_rich<SP>) : Base(ode, t0, q0, nsys, 0, 0, 0, inf<T>(), stepsize, dir, args) {}


template<typename T, size_t N, SolverPolicy SP, typename Derived>
Euler<T, N, SP, Derived>::Euler(OdeData<T> ode, T t0, const T* q0, size_t nsys, T stepsize, int dir, const std::vector<T>& args, EVENTS events) requires (is_rich<SP>) : Base(ode, t0, q0, nsys, 0, 0, 0, inf<T>(), stepsize, dir, args, events) {}


template<typename T, size_t N, SolverPolicy SP, typename Derived>
inline void Euler<T, N, SP, Derived>::interp_impl(T* result, const T& t) const{
    return lin_interp(result, t, this->t_old(), this->t_new(), this->old_state_ptr()+2, this->new_state_ptr()+2, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, typename Derived>
void Euler<T, N, SP, Derived>::adapt_impl(T* res){
    State<T> old = this->new_state();
    T& t_new = res[0];
    T& habs = res[1];
    T* vec = res+2;

    const T& stepsize = old.habs();
    t_new = old.t() + stepsize;
    habs = stepsize;

    this->rhs(vec, t_new, old.vector());
    for (size_t i=0; i<this->Nsys(); i++){
        // y2 = y1 + f*dt, f = dy/dt
        vec[i] = old.vector()[i] + vec[i]*stepsize;
    }
}

template<typename T, size_t N, SolverPolicy SP, typename Derived>
inline std::unique_ptr<Interpolator<T, N>> Euler<T, N, SP, Derived>::state_interpolator(int bdr1, int bdr2) const{
    return std::make_unique<LocalInterpolator<T, N>>(this->t_old(), this->t_new(), this->old_state_ptr()+2, this->new_state_ptr()+2, this->Nsys(), bdr1, bdr2);
}

#endif