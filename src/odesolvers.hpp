#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "events.hpp"
#include <limits>
#include <unordered_set>
#include "states.hpp"



#define MAIN_DEFAULT_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, const std::vector<T>& args={}, const std::vector<Event<T, N>*> events={}, const Functor<T, N>& mask=nullptr, std::string save_dir="", bool save_events_only=false

#define MAIN_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events, const Functor<T, N>& mask, std::string save_dir, bool save_events_only

#define SOLVER_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, T rtol, T atol, T min_step, T max_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events, const Functor<T, N>& mask, std::string save_dir, bool save_events_only, int err_est_ord, State<T, N>* initial_state

#define DERIVED_SOLVER_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, T rtol, T atol, T min_step, T max_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events, const Functor<T, N>& mask, std::string save_dir, bool save_events_only, int err_est_ord, STATE* initial_state

#define ODE_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, const std::vector<T>& args={}, std::string method="RK45", const std::vector<Event<T, N>*> events={}, const Functor<T, N>& mask=nullptr, std::string save_dir="", bool save_events_only=false

#define ARGS rhs, t0, q0, rtol, atol, min_step, max_step, first_step, args, events, mask, save_dir, save_events_only

#define PARTIAL_ARGS rhs, rtol, atol, min_step, max_step, args, events, mask, save_dir, save_events_only


template<class T, int N>
class OdeSolver;

template<class T, int N, class StateDerived>
class DerivedSolver;

template<class T, int N>
class OdeSolver{

    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

    template<class, int, class>
    friend class DerivedSolver;
    

public:

    const T MAX_FACTOR = T(10);
    const T SAFETY = T(9)/10;
    const T MIN_FACTOR = T(2)/10;
    const T ERR_EST_ORDER;


    virtual ~OdeSolver();


    void                            stop(const std::string& text = "");
    void                            kill(const std::string& text = "");
    bool                            advance();
    bool                            set_goal(const T& t_max);
    bool                            resume();
    bool                            release_file();
    bool                            reopen_file();
    bool                            free();

    virtual OdeRhs<T, N>            ode_rhs() const;
    inline const Functor<T, N>&     mask() const;
    inline const State<T, N>&       initial_state() const;
    inline const T&                 t() const;
    inline const vec<T, N>&         q() const;
    inline const vec<T, N>&         q_true() const;
    inline const T&                 stepsize() const;
    inline const T&                 tmax() const;
    inline const int&               direction() const;
    inline const T&                 rtol() const;
    inline const T&                 atol() const;
    inline const T&                 min_step() const;
    inline const T&                 max_step() const;
    inline const std::vector<T>&    args() const;
    inline const size_t&            Nsys() const;
    inline const bool&              diverges() const;
    inline const bool&              is_running() const;
    inline const bool&              is_dead() const;
    inline const vec<T, N>&         error()const;
    inline const std::string&       message();
    inline const bool&              autosave() const;
    inline bool                     file_is_ready() const;
    inline const std::string&       filename() const;
    inline const bool&              saves_events_only() const;
    inline const bool               at_event() const;
    inline std::string              event_name() const;
    inline const SolverState<T, N>  state() const;
    inline const Event<T, N>*       current_event() const;
    inline Event<T, N>*             current_event();
    inline const int&               current_event_index() const;
    inline const Event<T, N>*       event(const size_t& i)const;
    inline const Event<T, N>*const* event_array()const;
    inline size_t                   events_size() const;
    UniqueClone                     with_new_events(const std::vector<Event<T, N>*>& events) const;
    virtual OdeSolver<T, N>*        clone() const = 0;
    virtual UniqueClone             safe_clone() const = 0;
    T                               auto_step(T direction=0)const;
    const std::string&              name()const;

protected:

    OdeSolver(SOLVER_CONSTRUCTOR(T, N));

    OdeSolver(const OdeSolver<T, N>& other);

    OdeSolver(OdeSolver<T, N>&& other);

    OdeSolver<T, N>&    operator=(const OdeSolver<T, N>& other);
    
    const vec<T, N>&    _rhs(const T& t, const vec<T, N>& q) const;

    void                _rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const;
    
private:

    virtual void _adapt_impl() = 0;

    virtual bool _validate_state()=0;

    void _reverse_states();

    void _copy_data(const OdeSolver<T, N>& other);

    void _make_new_events(const std::vector<Event<T, N>*>& events);

    void _warn_dead();

    void _warn_paused();

    void _warn_travolta();

    void _delete_events();

    void _clear_states();


protected:
    //the two below, delete only in destructor, nowhere else
    //because derived classes might have their own pointers that
    //point to the same location, and deleting these two without
    //"notifying" the derived class will lead to undefined behavior
    State<T, N>* _state;
    State<T, N>* _old_state;
    State<T, N>* _initial_state;

private:

    Functor<T, N> _ode_rhs;
    T _rtol;
    T _atol;
    T _min_step;
    T _max_step;
    std::vector<T> _args;
    size_t _n; //size of ode system
    std::string _name;

    T _tmax;
    bool _diverges = false;
    bool _is_running = true;
    bool _is_dead = false;
    size_t _N=0;//total number of solution updates
    vec<T, N> _error;
    std::string _message; //different from "running".
    std::vector<Event<T, N>*> _events;
    int _current_event_index = -1;
    std::string _filename;
    std::ofstream _file;
    bool _autosave = false;
    bool _save_events_only = false;
    const vec<T, N>* _q_exposed = nullptr; //view_only pointer
    Functor<T, N> _mask = nullptr;
    mutable _MutableData<T, N, State<T, N>> _mut;
};


template<class T, int N, class STATE>
class DerivedSolver : public OdeSolver<T, N>{

protected:

    DerivedSolver(DERIVED_SOLVER_CONSTRUCTOR(T, N)) : OdeSolver<T, N>(name, PARTIAL_ARGS, err_est_ord, initial_state), _mut_der(*initial_state){}

    DerivedSolver(const DerivedSolver<T, N, STATE>& other) = default;

    DerivedSolver<T, N, STATE>& operator=(const DerivedSolver<T, N, STATE>& other){
        OdeSolver<T, N>::operator=(other);
        _copy_derived_data(other);
        return *this;
    }

    inline STATE* _state_ptr(){
        return static_cast<STATE*>(this->_state);
    }

    inline STATE* _old_state_ptr(){
        return static_cast<STATE*>(this->_old_state);
    }

    inline STATE* _initial_state_ptr(){
        return static_cast<STATE*>(this->_initial_state);
    }

private:

    virtual void _step_impl(STATE& result, const STATE& state, const T& h) const = 0; //h should carry the current sign

    inline void  _step_impl(STATE& result, const STATE& state, const T& h, const T& t_new)const; //t_new should be towards the current direction

    void         _partially_advance_to(const T& t_new);

    void         _partially_advance_by(const T& h);

    void         _partially_advance(const T& h, const T& t_new);

    bool         _adapt_to_event(Event<T, N>& event);

    bool         _validate_state() final;

    void         _copy_derived_data(const DerivedSolver<T, N, STATE>& other);


    Checkpoint<STATE> _checkpoint;
    _MutableData<T, N, STATE> _mut_der;

};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/

template<class T, int N>
OdeSolver<T, N>::~OdeSolver(){
    _clear_states();
    _delete_events();
    if (_autosave){
        _file.close();
    }
};

template<class T, int N>
OdeSolver<T, N>::UniqueClone OdeSolver<T, N>::with_new_events(const std::vector<Event<T, N>*>& events)const{
    UniqueClone new_solver = this->safe_clone();
    new_solver->_make_new_events(events);
    return new_solver;
}

template<class T, int N, class StateDerived>
inline void  DerivedSolver<T, N, StateDerived>::_step_impl(StateDerived& result, const StateDerived& state, const T& h, const T& t_new)const{ //t_new should be towards the current direction
    _step_impl(result, state, h);
    result._t = t_new;
}

template<class T, int N, class StateDerived>
void DerivedSolver<T, N, StateDerived>::_partially_advance_to(const T& t_new){
    _step_impl(*_state_ptr(), *_old_state_ptr(), t_new - _old_state_ptr()->t(), t_new);
}

template<class T, int N, class StateDerived>
void DerivedSolver<T, N, StateDerived>::_partially_advance_by(const T& h){
    _step_impl(*_state_ptr(), *_old_state_ptr(), h);
}

template<class T, int N, class StateDerived>
void DerivedSolver<T, N, StateDerived>::_partially_advance(const T& h, const T& t_new){
    _step_impl(*_state_ptr(), *_old_state_ptr(), h, t_new);
}

template<class T, int N, class StateDerived>
void DerivedSolver<T, N, StateDerived>::_copy_derived_data(const DerivedSolver<T, N, StateDerived>& other){
    *_initial_state_ptr() = *other._initial_state_ptr();
    *_old_state_ptr() = *other._old_state_ptr();
    *_state_ptr() = *other._state_ptr();
    _checkpoint = other._checkpoint;
}

template<class T, int N>
void OdeSolver<T, N>::_reverse_states(){
    //_old_state becomes _state, and _state becomes _old_state
    State<T, N>* old_tmp = _old_state;
    _old_state = _state;
    _state = old_tmp;
}

template<class T, int N>
void OdeSolver<T, N>::_clear_states(){
    delete _state;
    delete _old_state;
    delete _initial_state;
    _state = nullptr;
    _old_state = nullptr;
    _initial_state = nullptr;
}


template<class T, int N>
OdeRhs<T, N> OdeSolver<T, N>::ode_rhs() const { return {this->_ode_rhs, nullptr}; }

template<class T, int N>
inline const Functor<T, N>& OdeSolver<T, N>::mask() const { return _mask; }

template<class T, int N>
inline const State<T, N>& OdeSolver<T, N>::initial_state() const { return *_initial_state; }

template<class T, int N>
inline const T& OdeSolver<T, N>::t() const { return _state->t(); }

template<class T, int N>
inline const vec<T, N>& OdeSolver<T, N>::q() const {
    return *_q_exposed; }

template<class T, int N>
inline const vec<T, N>& OdeSolver<T, N>::q_true() const { return _state->vector(); }

template<class T, int N>
inline const T& OdeSolver<T, N>::stepsize() const { return _state->habs(); }

template<class T, int N>
inline const T& OdeSolver<T, N>::tmax() const { return _tmax; }

template<class T, int N>
inline const int& OdeSolver<T, N>::direction() const { return _state->direction(); }

template<class T, int N>
inline const T& OdeSolver<T, N>::rtol() const { return _rtol; }

template<class T, int N>
inline const T& OdeSolver<T, N>::atol() const { return _atol; }

template<class T, int N>
inline const T& OdeSolver<T, N>::min_step() const { return _min_step; }

template<class T, int N>
inline const T& OdeSolver<T, N>::max_step() const { return _max_step; }

template<class T, int N>
inline const std::vector<T>& OdeSolver<T, N>::args() const { return _args; }

template<class T, int N>
inline const size_t& OdeSolver<T, N>::Nsys() const { return _n; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::diverges() const { return _diverges; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::is_running() const { return _is_running; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::is_dead() const { return _is_dead; }

template<class T, int N>
inline const vec<T, N>& OdeSolver<T, N>::error() const { return _error; }

template<class T, int N>
inline const std::string& OdeSolver<T, N>::message() { return _message; }


template<class T, int N>
inline const std::string& OdeSolver<T, N>::name() const{ return _name; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::autosave() const { return _autosave; }

template<class T, int N>
T OdeSolver<T, N>::auto_step(T direction)const{
    //returns absolute value of emperically determined first step.
    const int dir = (direction == 0) ? _state->direction() : ( (direction > 0) ? 1 : -1);

    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    T h0, d2, h1;
    vec<T, N> y1, f1;
    vec<T, N> scale = _atol + _state->vector().cwiseAbs()*_rtol;
    vec<T, N> _dq = this->_rhs(_state->t(), _state->vector());

    T d0 = rms_norm((_state->vector()/scale).eval());
    T d1 = rms_norm((_dq/scale).eval());
    if (d0 < 1e-5 || d1 < 1e-5){
        h0 = 1e-6;
    }
    else{
        h0 = 0.01*d0/d1;
    }

    y1 = _state->vector()+h0*dir*_dq;
    f1 = _rhs(_state->t()+h0*dir, y1);

    d2 = rms_norm(((f1-_dq)/scale).eval()) / h0;
    
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = std::max(T(1e-6), 1e-3*h0);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }

    return std::min({100*h0, h1, this->_max_step});
}

template<class T, int N>
bool OdeSolver<T, N>::resume() {
    if (_is_dead){
        _warn_dead();
    }
    else if (_state->direction() == 0){
        _warn_travolta();
    }
    else{
        _message = "Running";
        _is_running = true;
        return true;
    }
    return false;
}

template<class T, int N>
bool OdeSolver<T, N>::release_file() {
    if (_autosave){
        _file.flush();
        _file.close();
        _autosave = false;
        return true;
    }
    else{
        return false;
    }
}

template<class T, int N>
inline bool OdeSolver<T, N>::file_is_ready() const {
    return _file.good();
}

template<class T, int N>
bool OdeSolver<T, N>::reopen_file() {
    if (_autosave || _filename.empty()){
        return false;
    }
    else{
        _file.open(_filename, std::ios::app);
        _autosave = true;
        return true;
    }
}

template<class T, int N>
inline const std::string& OdeSolver<T, N>::filename() const {
    return _filename;
}

template<class T, int N>
inline const bool& OdeSolver<T, N>::saves_events_only() const {
    return _save_events_only;
}

template<class T, int N>
bool OdeSolver<T, N>::free() {
    if (_state->direction() < 0){
        return set_goal(-inf<T>());
    }
    else{
        //default direction is positive
        return set_goal(inf<T>());
    }
}

template<class T, int N>
inline const bool OdeSolver<T, N>::at_event() const {
    return _current_event_index != -1;
}

template<class T, int N>
inline std::string OdeSolver<T, N>::event_name() const {
    return at_event() ? current_event()->name() : "";
}

template<class T, int N>
inline const SolverState<T, N> OdeSolver<T, N>::state() const {
    return {_state->t(), q(), _state->habs(), event_name(), _diverges, _is_running, _is_dead, _N, _message};
}

template<class T, int N>
inline const Event<T, N>* OdeSolver<T, N>::current_event() const {
    return at_event() ? _events[_current_event_index] : nullptr;
}

template<class T, int N>
inline Event<T, N>* OdeSolver<T, N>::current_event() {
    return at_event() ? _events[_current_event_index] : nullptr;
}

template<class T, int N>
inline const int& OdeSolver<T, N>::current_event_index() const {
    return _current_event_index;
}

template<class T, int N>
inline const Event<T, N>* OdeSolver<T, N>::event(const size_t& i)const{
    return _events[i];
}

template<class T, int N>
inline const Event<T, N>*const* OdeSolver<T, N>::event_array()const{
    return _events.data();
}

template<class T, int N>
inline size_t OdeSolver<T, N>::events_size() const {
    return _events.size();
}


template<class T, int N>
void OdeSolver<T, N>::stop(const std::string& text){
    _is_running = false;
    _message = (text == "") ? "Stopped by user" : text;
}



template<class T, int N>
void OdeSolver<T, N>::kill(const std::string& text){
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<class T, int N>
bool OdeSolver<T, N>::set_goal(const T& t_max_new){
    //if the solver was stopped (but not killed) earlier,
    //then setting a new goal successfully will resume the solver
    if (_is_dead){
        _warn_dead();
        return false;
    }
    else if (t_max_new == _state->t()){
        _state->set_direction(0);
        _tmax = t_max_new;
        stop("Waiting for new Tmax");
        return true;
    }
    else{
        _tmax = t_max_new;
        const T dir = t_max_new-_state->t();
        const T habs = _state->habs() == 0 ? this->auto_step(dir) : _state->habs();
        _state->adjust(habs, dir, this->_rhs(_state->t(), _state->vector()));
        return resume();
    }
}







template<typename T, int N>
OdeSolver<T, N>::OdeSolver(SOLVER_CONSTRUCTOR(T, N)): ERR_EST_ORDER(err_est_ord), _ode_rhs(rhs.ode_rhs), _rtol(rtol), _atol(atol), _min_step(min_step), _max_step(max_step), _args(args), _n(initial_state->vector().size()), _name(name), _error(initial_state->vector().size()), _filename(save_dir), _save_events_only(save_events_only), _mask(mask), _mut(*initial_state){
    std::unordered_set<std::string> seen;
    for (const Event<T, N>* ev : events) {
        if (!seen.insert(ev->name()).second) {
            throw std::runtime_error("Duplicate Event name not allowed: " + ev->name());
        }
    }
    if (_min_step < 0){
        throw std::runtime_error("Minimum stepsize must be a non negative number");
    }
    if (_max_step < _min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    initial_state->set_direction(0);
    //any "history" data in any implicit solvers will be overwritten anyway when a tmax goal is set,
    //because the direcion has been set to 0, so all that matters now is t0, q0, habs.
    _initial_state = initial_state;
    _initial_state->resize_step(1, min_step, max_step);
    _state = _initial_state->clone();
    _old_state = initial_state->clone();
    _q_exposed = &(_state->vector());
    _make_new_events(events);
    set_goal(_state->t());
    _error.setZero();

    if (!_filename.empty()){
        _file.open(_filename, std::ios::out);
        if (!_file){
            throw std::runtime_error("Could not open file in OdeSolver for automatic saving: " + _filename + "\n");
        }
        _autosave = true;
        write_checkpoint(_file, _state->t(), _state->vector(), -1);
    }
}

template<typename T, int N>
OdeSolver<T, N>::OdeSolver(const OdeSolver<T, N>& other):ERR_EST_ORDER(other.ERR_EST_ORDER){
    _initial_state = other._initial_state->clone();
    _state = other._state->clone();
    _old_state = other._old_state->clone();
    _copy_data(other);
    _filename = "";
    _autosave = false;

}

template<typename T, int N>
OdeSolver<T, N>::OdeSolver(OdeSolver<T, N>&& other):ERR_EST_ORDER(other.ERR_EST_ORDER), _state(other._state), _old_state(other._old_state), _initial_state(other._initial_state), _file(std::move(other._file)){
    // not the most efficient, but the most readable :)
    // besides the time scale of a copy is insignificant to the timescale of
    // solving an ode.
    _copy_data(other);
    other._state = nullptr;
    other._old_state = nullptr;
    other._initial_state = nullptr;

}

template<typename T, int N>
OdeSolver<T, N>& OdeSolver<T, N>::operator=(const OdeSolver<T, N>& other){
    _copy_data(other);
    _filename = "";
    _autosave = false;
    return *this;
}

template<typename T, int N>
inline const vec<T, N>& OdeSolver<T, N>::_rhs(const T& t, const vec<T, N>& q) const {
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

template<typename T, int N>
inline void OdeSolver<T, N>::_rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const{
    _ode_rhs(result, t, q, _args);
}


template<class T, int N>
bool OdeSolver<T, N>::advance(){
    this->_adapt_impl(); //_old_state changed, not _state
    if (_validate_state()){
        _error += _state->local_error();
        return true;
    }
    else{
        return false;
    }
}


template<class T, int N, class StateDerived>
bool DerivedSolver<T, N, StateDerived>::_adapt_to_event(Event<T, N>& event){
    //adapting to event must happend only after states have been reversed,
    //because _partiall_advance_to below changes _state wrt to _old_state,
    //and _old_state is the stable one, while _state still needs work.
    std::function<vec<T, N>(T)> qfunc = [this](const T& t){
        this->_step_impl(*_mut_der.state, *_old_state_ptr(), t-_old_state_ptr()->t());
        return _mut_der.state->vector();
    };
    
    if (event.determine(this->_old_state->t(), this->_old_state->vector(), this->_state->t(), this->_state->vector(), qfunc)){
        if (!this->at_event() && event.allows_checkpoint()){
            _checkpoint.set(*_state_ptr());
        }
        else if ( _checkpoint.is_set() && event.has_mask()){
            _checkpoint.remove();
        }
        _partially_advance_to(event.data().t()); //_state advances, not old state
        _state_ptr()->_q = event.data().q_true();
        if (event.hide_mask()){
            this->_q_exposed = &event.data().q();
        }
        return true;
    }
    return false;
}

template<class T, int N, class StateDerived>
bool DerivedSolver<T, N, StateDerived>::_validate_state(){
    //called immediately after any advance step that changed _old_state
    if (this->_is_dead){
        this->_warn_dead();
        return false;
    }
    else if (!this->_is_running){
        this->_warn_paused();
        return false;
    }
    
    //register state
    this->_reverse_states();
    this->_q_exposed = &this->_state->vector();
    if (_checkpoint.is_set()){
        *this->_state_ptr() = _checkpoint.state();
    }

    //change _state if any events are between _old_state and _state
    this->_current_event_index = -1;
    for (int i=0; i<static_cast<int>(this->_events.size()); i++){
        if (_adapt_to_event(*this->_events[i])){
            if (this->at_event()){
                this->current_event()->go_back();
            }
            this->_current_event_index = i;
            if (!this->current_event()->is_precise()){
                break;
            }
        }
    }

    //below we fall back to _old_state if needed
    bool success = true;
    if (!this->_state->vector().isFinite().all()){
        this->_reverse_states();
        this->kill("Ode solution diverges");
        this->_diverges = true;
        success = false;
    }
    else if (this->_state->habs() == 0){
        this->_reverse_states();
        this->kill("Required stepsize was smaller than machine precision");
        success = false;
    }

    //make or clear checkpoint first
    if (!this->at_event() && _checkpoint.is_set()){
        _checkpoint.remove();
    }

    if (this->_state->t()*this->_state->direction() >= this->_tmax*this->_state->direction()){

        if (this->_state->t()==this->_tmax){}
        else if (this->at_event()){
            //sometimes an event might appear a bit ahead of the tmax. This has already been registered
            //so we need to un-register it before stopping. It will be encoutered anyway when the solver is resumed.
            this->current_event()->go_back();
            this->_current_event_index = -1;
            this->_q_exposed = &this->_state->vector();
            this->_partially_advance_to(this->_tmax);
        }
        else{
            if (!this->_checkpoint.is_set()){
                this->_checkpoint.set(*_state_ptr());
            }
            this->_partially_advance_to(this->_tmax);
        }
        this->stop("T_max goal reached");
    }
    this->_N++;

    if (this->_mask != nullptr){
        this->_state->apply_mask(this->_mask, this->_args);
    }

    if (this->_autosave && success){
        if (!this->_save_events_only || this->at_event()){
            write_checkpoint(this->_file, this->_state->t(), this->q(), this->_current_event_index);
        }
    }

    if (this->at_event()){
        if (this->current_event()->is_leathal()){
            this->kill(this->current_event()->name());
        }
        else if (this->current_event()->is_stop_event()){
            this->stop(this->current_event()->name());
        }
    }
    return success;

}



template<typename T, int N>
inline void OdeSolver<T, N>::_warn_dead() {
    std::cout << std::endl << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << "\n";
}

template<typename T, int N>
inline void OdeSolver<T, N>::_warn_paused() {
    std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
}

template<typename T, int N>
inline void OdeSolver<T, N>::_warn_travolta() {
    std::cout << std::endl << "Solver has not been specified an integration direction, possibly because the Tmax goal was reached. Please set a new Tmax goal first or free() the solver.\n";
}

template<typename T, int N>
void OdeSolver<T, N>::_copy_data(const OdeSolver<T, N>& other) {
    //does not copy _file, this has to be managed outside this function
    //checkpoint and states are handled by DerivedSolver
    /*
    _copy_data must be organized to always be called with DerivedSolver::_copy_derived_data
    This does not need any special handling, but is naturally happening since a proper copy
    always calls the copy constructor/assignment operator of the derived classes
    */

    _ode_rhs = other._ode_rhs;
    _rtol = other._rtol;
    _atol = other._atol;
    _min_step = other._min_step;
    _max_step = other._max_step;
    _args = other._args;
    _n = other._n;
    _name = other._name;

    _tmax = other._tmax;
    _diverges = other._diverges;
    _is_running = other._is_running;
    _is_dead = other._is_dead;
    _N = other._N;
    _error = other._error;
    _message = other._message;
    _current_event_index = other._current_event_index;
    _save_events_only = other._save_events_only;

    _filename = other._filename;
    _autosave = other._autosave;
    _mask = other._mask;
    _make_new_events(other._events);
    if (other._q_exposed == &other._state->vector()){
        _q_exposed = &_state->vector();
    }
    else{
        _q_exposed = &current_event()->data().q();
    }
    _mut = other._mut;
}


template<typename T, int N>
void OdeSolver<T, N>::_make_new_events(const std::vector<Event<T, N>*>& events) {

    // FIRST create a new vector with new allocated objects, because "events" might be
    // our current _events vector. We sort the vector to contain normal events first,
    // and stop_events after to improve runtime performance and not miss out on any stop_events
    // if a single step encounters multiple events.
    std::vector<Event<T, N>*> new_precise_events;
    std::vector<Event<T, N>*> new_rough_events;
    for (size_t i = 0; i < events.size(); i++) {
        if (events[i]->is_precise()) {
            new_precise_events.push_back(events[i]->clone());
        }
        else {
            new_rough_events.push_back(events[i]->clone());
        }
    }

    // push the pointers into a new (sorted) array
    std::vector<Event<T, N>*> result(events.size());
    std::copy(new_precise_events.begin(), new_precise_events.end(), result.begin());
    std::copy(new_rough_events.begin(), new_rough_events.end(), result.begin() + new_precise_events.size());

    for (Event<T, N>* ev : result) {
        ev->set_args(this->_args);
    }

    // NOW we can delete our current events
    _delete_events();

    _events = result;
}

template<typename T, int N>
void OdeSolver<T, N>::_delete_events() {
    for (size_t i = 0; i < _events.size(); i++) {
        delete this->_events[i];
        this->_events[i] = nullptr;
    }
}






#endif



