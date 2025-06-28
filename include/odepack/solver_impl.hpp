#ifndef SOLVER_IMPL_HPP
#define SOLVER_IMPL_HPP

#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include "events.hpp"
#include "odesolvers.hpp"
#include "states.hpp"


template<typename T, int N, class Derived, class STATE>
class DerivedAdaptiveStepSolver;

template<typename T, int N, class Derived, class STATE>
class DerivedFixedStepSolver;


template<typename T, int N, class Derived, class STATE>
class DerivedSolver : public OdeSolver<T, N>{

    using Solver = DerivedSolver<T, N, Derived, STATE>;
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

    template<typename A, int B, class C, class D>
    friend class DerivedAdaptiveStepSolver;

    template<typename A, int B, class C, class D>
    friend class DerivedFixedStepSolver;

public:

    DerivedSolver() = delete;

    ~DerivedSolver();

    OdeRhs<T, N>                 ode_rhs() const override;
    const T&                     t() const final;
    const vec<T, N>&             q() const final;
    const vec<T, N>&             q_true() const final;
    const T&                     stepsize() const final;
    const T&                     tmax() const final;
    const int&                   direction() const final;
    const std::vector<T>&        args() const final;
    const size_t&                Nsys() const final;
    const bool&                  diverges() const final;
    const bool&                  is_running() const final;
    const bool&                  is_dead() const final;
    const std::string&           message() const final;
    const SolverState<T, N>      state() const final;
    const EventCollection<T, N>& events() const final;
    bool                         at_event() const final;
    std::string                  event_name() const final;
    const Event<T, N>&           current_event() const final;
    const int&                   current_event_index() const final;
    const std::string&           name() const final;
    OdeSolver<T, N>*             clone() const final;
    UniqueClone                  safe_clone() const final;
    UniqueClone                  with_new_events(const EventCollection<T, N>& events) const final;
    Derived*                     derived_clone() const;

    bool                         advance() final;
    bool                         set_goal(const T& t_max) final;
    void                         stop(const std::string& text) final;
    void                         kill(const std::string& text) final;
    bool                         resume() final;
    bool                         free() final;

    void                         set_goal_impl(const T& tmax_new);//virtual

    STATE                        new_state(const T& t, const vec<T, N>& q, const T& h) const;//virtual

    bool                         advance_impl();//virtual

    void                         call_impl(vec<T, N>& res, const T& t, const State<T, N>& state1, const State<T, N>& state2) const;//virtual

protected:

    DerivedSolver(const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, const std::vector<T>& args, const std::vector<Event<T, N>*> events);

    DerivedSolver(const DerivedSolver& other);

    DerivedSolver(DerivedSolver&& other);

    DerivedSolver& operator=(const DerivedSolver& other);

    DerivedSolver& operator=(DerivedSolver&& other);

    const Functor<T, N>& _rhs() const;
    
    const vec<T, N>&     _rhs(const T& t, const vec<T, N>& q) const;

    void                 _rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const;

    const vec<T, N>&     _interp(const T& t, const State<T, N>& state1, const State<T, N>& state2) const;

    void                 _finalize(const T& t0, const vec<T, N>& q0, T first_step);

private:

    Event<T, N>& current_event();

    void         _register_states();

    bool         _adapt_to_event(Event<T, N>& event, const State<T, N>& before, const State<T, N>& after);

    bool         _validate_it(const STATE& state);

    void         _warn_dead();

    void         _warn_paused();

    void         _warn_travolta();

    void         _find_true_state(const Solver& other);

    void         _finalize_state(const State<T, N>& start);

    void         _clear_states();

    void         _copy_data(const Solver& other);

    void         _move_data(Solver&& other);


    Functor<T, N>              _ode_rhs;
    std::vector<T>             _args;
    size_t                     _n; //size of ode system
    std::string                _name;
    T                          _tmax;
    bool                       _diverges = false;
    bool                       _is_running = true;
    bool                       _is_dead = false;
    size_t                     _N=0;//total number of solution updates
    std::string                _message;
    EventCollection<T, N>      _events;
    int                        _current_event_index = -1;
    int                        _direction;
    mutable _MutableData<T, N> _mut;
    STATE*                     _initial_state;
    STATE*                     _state;
    STATE*                     _old_state;
    STATE*                     _aux_state;
    ViewState<T, N>*           _temp_state; //mostly in case when tmax is met
    const State<T, N>*         _true_state;
    bool                       _equiv_states = true;

};



/*
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
------------------------------------------------IMPLEMENTATIONS-------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
*/





template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>::DerivedSolver(const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, const std::vector<T>& args, const std::vector<Event<T, N>*> events): OdeSolver<T, N>(), _ode_rhs(rhs.ode_rhs), _args(args), _n(q0.size()), _name(name), _events(events), _mut({q0, q0}){
    std::unordered_set<std::string> seen;
    for (const Event<T, N>* ev : events) {
        if (!seen.insert(ev->name()).second) {
            throw std::runtime_error("Duplicate Event name not allowed: " + ev->name());
        }
    }

    _events.set_args(args);
}



template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>::DerivedSolver(const DerivedSolver& other) : OdeSolver<T, N>(other), _ode_rhs(other._ode_rhs), _args(other._args), _n(other._n), _name(other._name), _tmax(other._tmax), _diverges(other._diverges), _is_running(other._is_running), _is_dead(other._is_dead), _N(other._N), _message(other._message), _events(other._events), _current_event_index(other._current_event_index), _direction(other._direction), _mut(other._mut), _equiv_states(other._equiv_states){
    _initial_state = new STATE(*other._initial_state);
    _state = new STATE(*other._state);
    _old_state = new STATE(*other._old_state);
    _aux_state = new STATE(*other._aux_state);
    _temp_state = new ViewState(*other._temp_state);
    _find_true_state(other);
}


template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>::DerivedSolver(DerivedSolver&& other) : OdeSolver<T, N>(std::move(other)), _ode_rhs(std::move(other._ode_rhs)), _args(std::move(other._args)), _n(std::move(other._n)), _name(std::move(other._name)), _tmax(std::move(other._tmax)), _diverges(std::move(other._diverges)), _is_running(std::move(other._is_running)), _is_dead(std::move(other._is_dead)), _N(std::move(other._N)), _message(std::move(other._message)), _events(std::move(other._events)), _current_event_index(std::move(other._current_event_index)), _direction(std::move(other._direction)), _mut(std::move(other._mut)), _initial_state(other._initial_state), _state(other._state), _old_state(other._old_state), _aux_state(other._aux_state), _temp_state(other._temp_state), _equiv_states(std::move(other._equiv_states)){
    _find_true_state(other);
    other._initial_state = nullptr;
    other._state = nullptr;
    other._old_state = nullptr;
    other._aux_state = nullptr;
    other._temp_state = nullptr;
}

template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>& DerivedSolver<T, N, Derived, STATE>::operator=(const DerivedSolver& other){
    OdeSolver<T, N>::operator=(other);
    _copy_data(other);
    return *this;
}


template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>& DerivedSolver<T, N, Derived, STATE>::operator=(DerivedSolver&& other){
    OdeSolver<T, N>::operator=(std::move(other));
    _move_data(std::move(other));
    return *this;
}











template<typename T, int N, class Derived, class STATE>
DerivedSolver<T, N, Derived, STATE>::~DerivedSolver() {
    _clear_states();
}



template<typename T, int N, class Derived, class STATE>
inline OdeRhs<T, N> DerivedSolver<T, N, Derived, STATE>::ode_rhs() const {
    return {this->_ode_rhs, nullptr};
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedSolver<T, N, Derived, STATE>::t() const {
    return _true_state->t();
}

template<typename T, int N, class Derived, class STATE>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE>::q() const {
    return _true_state->exposed_vector();
}

template<typename T, int N, class Derived, class STATE>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE>::q_true() const {
    return _true_state->vector();
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedSolver<T, N, Derived, STATE>::stepsize() const {
    return _state->habs();
}

template<typename T, int N, class Derived, class STATE>
inline const T& DerivedSolver<T, N, Derived, STATE>::tmax() const {
    return _tmax;
}

template<typename T, int N, class Derived, class STATE>
inline const int& DerivedSolver<T, N, Derived, STATE>::direction() const {
    return _direction;
}

template<typename T, int N, class Derived, class STATE>
inline const std::vector<T>& DerivedSolver<T, N, Derived, STATE>::args() const {
    return _args;
}

template<typename T, int N, class Derived, class STATE>
inline const size_t& DerivedSolver<T, N, Derived, STATE>::Nsys() const {
    return _n;
}

template<typename T, int N, class Derived, class STATE>
inline const bool& DerivedSolver<T, N, Derived, STATE>::diverges() const {
    return _diverges;
}

template<typename T, int N, class Derived, class STATE>
inline const bool& DerivedSolver<T, N, Derived, STATE>::is_running() const {
    return _is_running;
}

template<typename T, int N, class Derived, class STATE>
inline const bool& DerivedSolver<T, N, Derived, STATE>::is_dead() const {
    return _is_dead;
}

template<typename T, int N, class Derived, class STATE>
inline const std::string& DerivedSolver<T, N, Derived, STATE>::message() const {
    return _message;
}

template<typename T, int N, class Derived, class STATE>
inline const SolverState<T, N> DerivedSolver<T, N, Derived, STATE>::state() const {
    return {_true_state->t(), _true_state->exposed_vector(), _state->habs(), event_name(), _diverges, _is_running, _is_dead, _N, _message};
}

template<typename T, int N, class Derived, class STATE>
inline const EventCollection<T, N>& DerivedSolver<T, N, Derived, STATE>::events() const {
    return _events;
}

template<typename T, int N, class Derived, class STATE>
inline bool DerivedSolver<T, N, Derived, STATE>::at_event() const {
    return _current_event_index != -1;
}

template<typename T, int N, class Derived, class STATE>
inline std::string DerivedSolver<T, N, Derived, STATE>::event_name() const {
    return at_event() ? current_event().name() : "";
}

template<typename T, int N, class Derived, class STATE>
inline const Event<T, N>& DerivedSolver<T, N, Derived, STATE>::current_event() const {
    return _events[_current_event_index];
}

template<typename T, int N, class Derived, class STATE>
inline Event<T, N>& DerivedSolver<T, N, Derived, STATE>::current_event() {
    return _events[_current_event_index];
}

template<typename T, int N, class Derived, class STATE>
inline const int& DerivedSolver<T, N, Derived, STATE>::current_event_index() const {
    return _current_event_index;
}

template<typename T, int N, class Derived, class STATE>
inline const std::string& DerivedSolver<T, N, Derived, STATE>::name() const {
    return _name;
}


template<typename T, int N, class Derived, class STATE>
inline OdeSolver<T, N>* DerivedSolver<T, N, Derived, STATE>::clone()const{
    return new Derived(*static_cast<const Derived*>(this));
}

template<typename T, int N, class Derived, class STATE>
inline std::unique_ptr<OdeSolver<T, N>> DerivedSolver<T, N, Derived, STATE>::safe_clone()const{
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
}


template<typename T, int N, class Derived, class STATE>
inline std::unique_ptr<OdeSolver<T, N>> DerivedSolver<T, N, Derived, STATE>::with_new_events(const EventCollection<T, N>& events)const{
    Derived* cl = derived_clone();
    cl->_events = events;
    cl->_events.set_args(this->_args);
    return std::unique_ptr<OdeSolver<T, N>>(cl);
}


template<typename T, int N, class Derived, class STATE>
inline Derived* DerivedSolver<T, N, Derived, STATE>::derived_clone()const{
    return new Derived(*static_cast<const Derived*>(this));
}


template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::advance(){
    return this->advance_impl();
}


template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::set_goal(const T& t_max_new){
    //if the solver was stopped (but not killed) earlier,
    //then setting a new goal successfully will resume the solver
    if (this->_is_dead){
        this->_warn_dead();
        return false;
    }
    else if (t_max_new == _true_state->t()){
        _direction = 0;
        this->_tmax = t_max_new;
        stop("Waiting for new Tmax");
        return true;
    }
    else{
        this->set_goal_impl(t_max_new);
        return resume();
    }
}



template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::stop(const std::string& text){
    _is_running = false;
    _message = (text == "") ? "Stopped by user" : text;
}


template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::kill(const std::string& text){
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::resume(){
    if (_is_dead){
        _warn_dead();
    }
    else if (_direction == 0){
        _warn_travolta();
    }
    else{
        _message = "Running";
        _is_running = true;
        return true;
    }
    return false;
}


template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::free() {
    if (_direction < 0){
        return set_goal(-inf<T>());
    }
    else{
        //default direction is positive
        return set_goal(inf<T>());
    }
}




template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::set_goal_impl(const T& tmax_new) {
    return static_cast<Derived*>(this)->set_goal_impl(tmax_new);
}

template<typename T, int N, class Derived, class STATE>
STATE DerivedSolver<T, N, Derived, STATE>::new_state(const T& t, const vec<T, N>& q, const T& h) const {
    return static_cast<const Derived*>(this)->new_state(t, q, h);
}

template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::advance_impl() {
    return static_cast<Derived*>(this)->advance_impl();
}


template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::call_impl(vec<T, N>& res, const T& t, const State<T, N>& state1, const State<T, N>& state2) const{
    return static_cast<const Derived*>(this)->call_impl(res, t, state1, state2);
}










template<typename T, int N, class Derived, class STATE>
inline const Functor<T, N>& DerivedSolver<T, N, Derived, STATE>::_rhs() const{
    return _ode_rhs;
}



template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const{
    _ode_rhs(result, t, q, _args);
}


template<typename T, int N, class Derived, class STATE>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE>::_rhs(const T& t, const vec<T, N>& q) const {
    /*
    only assign a copy:
        e.g. vec<T, N> r = this->_rhs(...); allowed

        do NOT do this:
            const vec<T, N>& r = this->_rhs(...);

    This is because a reference to a mutable object is returned
    to improve efficiency. If the function is called with different arguments,
    the underlying mutable object will change value, which will lead to unexpected behavior
    when using the first const& defined.
    */
    _ode_rhs(_mut.qdiff, t, q, _args);
    return _mut.qdiff;
}


template<typename T, int N, class Derived, class STATE>
const vec<T, N>& DerivedSolver<T, N, Derived, STATE>::_interp(const T& t, const State<T, N>& state1, const State<T, N>& state2) const{
    this->call_impl(_mut.q, t, state1, state2);
    return _mut.q;
}


template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::_finalize(const T& t0, const vec<T, N>& q0, T first_step){
    _initial_state = new STATE(new_state(t0, q0, first_step));
    _state = new STATE(*_initial_state);
    _old_state = new STATE(*_initial_state);
    _aux_state = new STATE(*_initial_state);
    _temp_state = new ViewState<T, N>(t0, q0);
    _true_state = _state;
}








template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_register_states(){
    STATE* tmp = _state;
    _state = _aux_state;
    _aux_state = _old_state;
    _old_state = tmp;
    _true_state = _state; //default value of _true_state, but it might change before the step is finalized
    _equiv_states = true;
}


template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::_adapt_to_event(Event<T, N>& event, const State<T, N>& before, const State<T, N>& after){

    //MUST NOT change _state or _old_state in here, only _true_state

    std::function<vec<T, N>(T)> qfunc = [this](const T& t){
        return this->_interp(t, *this->_old_state, *this->_state);
    };
    
    if (event.determine(before.t(), before.vector(), after.t(), after.vector(), qfunc)){
        _true_state = &event.state();
        _equiv_states = false;
        return true;
    }
    return false;
}


template<typename T, int N, class Derived, class STATE>
bool DerivedSolver<T, N, Derived, STATE>::_validate_it(const STATE& state){
    bool success = true;
    if (!state.vector().isFinite().all()){
        this->kill("Ode solution diverges");
        this->_diverges = true;
        success = false;
    }
    else if (state.habs() == 0){
        this->kill("Required stepsize was smaller than machine precision");
        success = false;
    }

    return success;
}

template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_warn_dead() {
    std::cout << std::endl << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << "\n";
}

template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_warn_paused() {
    std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
}

template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_warn_travolta() {
    std::cout << std::endl << "Solver has not been specified an integration direction, possibly because the Tmax goal was reached. Please set a new Tmax goal first or free() the solver.\n";
}


template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_find_true_state(const Solver& other){
    if (other._true_state == other._state){
        _true_state = _state;
        return;
    }
    else if(other._true_state == other._temp_state){
        _true_state = _temp_state;
        return;
    }
    else if (other.at_event() && (other._true_state != &other.current_event().state())){
        throw std::runtime_error("True state not found in copy.");
    }
    _true_state = &current_event().state();
}


template<typename T, int N, class Derived, class STATE>
void DerivedSolver<T, N, Derived, STATE>::_finalize_state(const State<T, N>& start){

    this->_current_event_index = -1;
    
    for (int i=0; i<static_cast<int>(this->_events.size()); i++){
        //_true_state dynamically changes below if many events are encoutered in a single step.
        if (_adapt_to_event(this->_events[i], start, *_true_state)){
            if (this->at_event()){
                this->current_event().go_back();
            }
            this->_current_event_index = i;
            if (!this->current_event().is_precise()){
                break;
            }
        }
    }

    if (_true_state->t()*_state->direction() >= this->_tmax*_state->direction()){

        if (_true_state->t()==_tmax){}
        else if (this->at_event()){
            //sometimes an event might appear a bit ahead of the tmax. This has already been registered
            //so we need to un-register it before stopping. It will be encoutered anyway when the solver is resumed.

            this->current_event().go_back();
            this->_current_event_index = -1;
            //if the event was masked, then _true_state is pointing at _temp_state.
            //Since the tmax is before the masked event, the interpolating coefficients can still be used
            //even if they range from _old_state to _state, where _state appears *after* the event.
            *_temp_state = std::move(ViewState<T, N>(_tmax, _interp(_tmax, *_old_state, *_state)));
            _true_state = _temp_state;
            _equiv_states = false;
        }
        else{
            *_temp_state = std::move(ViewState<T, N>(_tmax, _interp(_tmax, *_old_state, *_state)));
            _true_state = _temp_state;
            _equiv_states = false;
        }
        this->stop("T_max goal reached");
    }
    else if (this->at_event()){

        if (!this->current_event().allows_checkpoint()){
            //make new start. Interpolating coefficients might not be valid until the next step.
            *_state = std::move(this->new_state(_true_state->t(), _true_state->vector(), _state->h()));
            _equiv_states = true;
            //do not set _true_state = _state yet, because if the event has a hidden mask,
            //the event state exposed vector will not be viewed.
        }

        if (this->current_event().is_leathal()){
            this->kill(this->current_event().name());
        }
        else if (this->current_event().is_stop_event()){
            this->stop(this->current_event().name());
        }
    }


    this->_N++;
}


template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_clear_states(){
    delete _initial_state;
    delete _state;
    delete _old_state;
    delete _aux_state;
    delete _temp_state;
    _initial_state = nullptr;
    _state = nullptr;
    _old_state = nullptr;
    _aux_state = nullptr;
    _temp_state = nullptr;
    _true_state = nullptr;
}


template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_copy_data(const Solver& other){
    _ode_rhs = other._ode_rhs;
    _args = other._args;
    _n = other._n;
    _name = other._name;
    _tmax = other._tmax;
    _diverges = other._diverges;
    _is_running = other._is_running;
    _is_dead = other._is_dead;
    _N = other._N;
    _message = other._message;
    _events = other._events;
    _current_event_index = other._current_event_index;
    _direction = other._direction;
    _mut = other._mut;
    _equiv_states = other._equiv_states;

    *_initial_state = *other._initial_state;
    *_state = *other._state;
    *_old_state = *other._old_state;
    *_aux_state = *other._aux_state;
    *_temp_state = *other._temp_state;
    _find_true_state(other);
}


template<typename T, int N, class Derived, class STATE>
inline void DerivedSolver<T, N, Derived, STATE>::_move_data(Solver&& other) {
    _ode_rhs = std::move(other._ode_rhs);
    _args = std::move(other._args);
    _n = std::move(other._n);
    _name = std::move(other._name);
    _tmax = std::move(other._tmax);
    _diverges = std::move(other._diverges);
    _is_running = std::move(other._is_running);
    _is_dead = std::move(other._is_dead);
    _N = std::move(other._N);
    _message = std::move(other._message);
    _events = std::move(other._events);
    _current_event_index = std::move(other._current_event_index);
    _direction = std::move(other._direction);
    _mut = std::move(other._mut);
    _equiv_states = std::move(other._equiv_states);

    *_initial_state = std::move(*other._initial_state);
    *_state = std::move(*other._state);
    *_old_state = std::move(*other._old_state);
    *_aux_state = std::move(*other._aux_state);
    *_temp_state = std::move(*other._temp_state);

    other._initial_state = nullptr;
    other._state = nullptr;
    other._old_state = nullptr;
    other._aux_state = nullptr;
    other._temp_state = nullptr;
    other._true_state = nullptr;
}

#endif