#ifndef ADAPTIVE_SOLVER_IMPL_HPP
#define ADAPTIVE_SOLVER_IMPL_HPP

#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include "events.hpp"
#include "solver_impl.hpp"
#include "states.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, const std::vector<T>& args={}, const std::vector<Event<T, N>*> events={}

#define SOLVER_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events

#define ODE_CONSTRUCTOR(T, N) MAIN_DEFAULT_CONSTRUCTOR(T, N), std::string method="RK45"

#define ARGS rhs, t0, q0, rtol, atol, min_step, max_step, first_step, args, events


template<typename T, int N, class Derived, class STATE>
class DerivedAdaptiveStepSolver : public DerivedSolver<T, N, Derived, STATE>{

    //TODO: AdaptiveOdeSolver class for abstract interface.
    //Then, this class will perform a diamond inheritance, inheriting from AdaptiveSolver and DerivedSolver.

public:

    static const T MAX_FACTOR;
    static const T SAFETY;
    static const T MIN_FACTOR;

    DerivedAdaptiveStepSolver() = delete;


    const T&                     rtol() const;
    const T&                     atol() const;
    const T&                     min_step() const;
    const T&                     max_step() const;
    const vec<T, N>&             error() const;
    T                            auto_step(T direction=0, const ICS<T, N>* = nullptr) const;

    bool                         advance_impl();
    void                         set_goal_impl(const T& tmax_new);

    void                         adapt_impl(STATE& res, const STATE& state);//virtual

protected:

    DerivedAdaptiveStepSolver(SOLVER_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(DerivedAdaptiveStepSolver);

    void _finalize(const T& t0, const vec<T, N>& q0, T first_step);

private:

    T                          _rtol;
    T                          _atol;
    T                          _min_step;
    T                          _max_step;
    vec<T, N>                  _error;

};



/*
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
------------------------------------------------IMPLEMENTATIONS-------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
*/





template<typename T, int N, class Derived, class STATE>
DerivedAdaptiveStepSolver<T, N, Derived, STATE>::DerivedAdaptiveStepSolver(SOLVER_CONSTRUCTOR(T, N)): DerivedSolver<T, N, Derived, STATE>(name, rhs, t0, q0, args, events), _rtol(rtol), _atol(atol), _min_step(min_step), _max_step(max_step), _error(q0.size()){
    if (min_step < 0){
        throw std::runtime_error("Minimum stepsize must be a non negative number");
    }
    if (max_step < min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    _error.setZero();
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedAdaptiveStepSolver<T, N, Derived, STATE>::rtol() const {
    return _rtol;
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedAdaptiveStepSolver<T, N, Derived, STATE>::atol() const {
    return _atol;
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedAdaptiveStepSolver<T, N, Derived, STATE>::min_step() const {
    return _min_step;
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedAdaptiveStepSolver<T, N, Derived, STATE>::max_step() const {
    return _max_step;
}

template<typename T, int N, class Derived, class STATE>
inline const vec<T, N>& DerivedAdaptiveStepSolver<T, N, Derived, STATE>::error() const {
    return _error;
}

template<typename T, int N, class Derived, class STATE>
T DerivedAdaptiveStepSolver<T, N, Derived, STATE>::auto_step(T direction, const ICS<T, N>* ics)const{
    //returns absolute value of emperically determined first step.
    const int dir = (direction == 0) ? this->_state->direction() : ( (direction > 0) ? 1 : -1);
    const T& t = (ics == nullptr) ? this->_state->t() : ics->t;
    const vec<T, N>& q = (ics == nullptr) ? this->_state->vector() : ics->q;

    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    T h0, d2, h1;
    vec<T, N> y1, f1;
    vec<T, N> scale = _atol + q.cwiseAbs()*_rtol;
    vec<T, N> _dq = this->_rhs(t, q);

    T d0 = rms_norm((q/scale).eval());
    T d1 = rms_norm((_dq/scale).eval());
    if (d0 * 100000 < 1 || d1 * 100000 < 1){
        h0 = T(1)/1000000;
    }
    else{
        h0 = d0/d1/100;
    }

    y1 = q+h0*dir*_dq;
    f1 = this->_rhs(t+h0*dir, y1);

    d2 = rms_norm(((f1-_dq)/scale).eval()) / h0;
    
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = std::max(T(1)/1000000, h0/1000);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(Derived::ERR_EST_ORDER+1));
    }

    return std::max(std::min({100*h0, h1, this->_max_step}), this->_min_step);
}

template<typename T, int N, class Derived, class STATE>
bool DerivedAdaptiveStepSolver<T, N, Derived, STATE>::advance_impl(){

    if (this->_is_dead){
        this->_warn_dead();
        return false;
    }
    else if (!this->_is_running){
        this->_warn_paused();
        return false;
    }


    if (this->_equiv_states){
        this->adapt_impl(*this->_aux_state, *this->_state); //only *_aux_state changed
        if (this->_validate_it(*this->_aux_state)){
            this->_register_states(); //now _old_state became _state, _state is updated, and _true_state points to _state
            this->_finalize_state(*this->_old_state); //ONLY affects the _true_state pointer
            this->_error += this->_true_state->local_error();
        }
        else{
            return false;
        }
    }
    else{
        const State<T, N>* tmp = this->_true_state;
        this->_true_state = this->_state; //temporarily set to the next naturally adapted state.
        this->_equiv_states = true;
        this->_finalize_state(*tmp);
    }
    return true;
}


template<typename T, int N, class Derived, class STATE>
void DerivedAdaptiveStepSolver<T, N, Derived, STATE>::set_goal_impl(const T& t_max_new){
    this->_tmax = t_max_new;
    const T dir = t_max_new-this->_true_state->t();
    if (dir*this->_state->direction() < 0){
        this->_direction = sgn(dir);
        this->_state->adjust(auto_step(dir), dir, this->_rhs(this->_state->t(), this->_state->vector()));
        
        //_true_state might lie somewhere between _old_state and _state.
        //So _state and _old_state must accordinly adapt so that _true_state still lies between them,
        //but only one step difference (the step difference might have changed when we changed direction)
        if (!this->_equiv_states){
            //the interior *_true_state will not be affected below, even if the _true_state pointer itself will.
            const State<T, N>* const tmp = this->_true_state; //_true_state pointer will be affected below, so we need to store it first.
            while (this->_state->t()*dir < tmp->t()*dir){
                adapt_impl(*this->_aux_state, *this->_state);
                this->_register_states();
            }
            this->_true_state = tmp;
        }
    }
}


template<typename T, int N, class Derived, class STATE>
void DerivedAdaptiveStepSolver<T, N, Derived, STATE>::adapt_impl(STATE& res, const STATE& state){
    return static_cast<Derived*>(this)->adapt_impl(res, state);
}

template<typename T, int N, class Derived, class STATE>
void DerivedAdaptiveStepSolver<T, N, Derived, STATE>::_finalize(const T& t0, const vec<T, N>& q0, T first_step){
    int dir = sgn(first_step);

    if (first_step != 0){
        first_step = choose_step(abs(first_step), _min_step, _max_step);
    }
    else{
        const ICS<T, N> ics = {t0, q0};
        first_step = this->auto_step(1, &ics);
        dir = 1;
    }
    //now first_step and initial direction are both != 0.
    this->_direction = dir;
    DerivedSolver<T, N, Derived, STATE>::_finalize(t0, q0, first_step*dir);
}


template<typename T, int N, class Derived, class STATE>
const T DerivedAdaptiveStepSolver<T, N, Derived, STATE>::MAX_FACTOR = T(10);

template<typename T, int N, class Derived, class STATE>
const T DerivedAdaptiveStepSolver<T, N, Derived, STATE>::SAFETY = T(9)/10;

template<typename T, int N, class Derived, class STATE>
const T DerivedAdaptiveStepSolver<T, N, Derived, STATE>::MIN_FACTOR = T(2)/10;


#endif