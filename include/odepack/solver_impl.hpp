#ifndef SOLVER_IMPL_HPP
#define SOLVER_IMPL_HPP

#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include "events.hpp"
#include "odesolvers.hpp"
#include "interpolators.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, const std::vector<T>& args={}, const std::vector<Event<T, N>*> events={}

#define MAIN_CONSTRUCTOR(T, N) const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events

#define SOLVER_CONSTRUCTOR(T, N) const std::string& name, const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, const std::vector<T>& args, const std::vector<Event<T, N>*> events

#define ODE_CONSTRUCTOR(T, N) MAIN_DEFAULT_CONSTRUCTOR(T, N), std::string method="RK45"

#define ARGS rhs, t0, q0, rtol, atol, min_step, max_step, first_step, args, events

#define PARTIAL_ARGS rhs, rtol, atol, min_step, max_step, args, events


template<typename T, int N, typename INTERPOLATOR>
struct _MutableData{

    vec<T, N>       q;
    vec<T, N>       qdiff;
    INTERPOLATOR    interpolator;
    Eigen::Matrix<T, N, -1> coef_mat;
    bool            interpolator_is_set = false;

    _MutableData(const vec<T, N>& q, int interp_order) : q(q), qdiff(q), interpolator(0, q), coef_mat(q.size(), interp_order) {}
};

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
class DerivedSolver : public OdeSolver<T, N>{

    using Solver = DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>;
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;
    using InterpolatorList = std::vector<const Interpolator<T, N>*>;

    static constexpr bool _req_coef_mat = std::is_same_v<INTERPOLATOR, StandardLocalInterpolator<T, N, STATE>>;

public:

    static const T MAX_FACTOR;
    static const T SAFETY;
    static const T MIN_FACTOR;

    static constexpr int INTERP_ORDER = []() constexpr {
        if constexpr (_req_coef_mat) {
            return Derived::INTERP_ORDER; //virtual. required if _req_coef_mat is true
        }
        else {
            return 0;
        }
    }();

    DerivedSolver() = delete;

    ~DerivedSolver();

    OdeRhs<T, N>                    ode_rhs() const override;
    const T&                        t() const final;
    const vec<T, N>&                q() const final;
    const vec<T, N>&                q_true() const final;
    const T&                        stepsize() const final;
    const T&                        tmax() const final;
    const int&                      direction() const final;
    const T&                        rtol() const final;
    const T&                        atol() const final;
    const T&                        min_step() const final;
    const T&                        max_step() const final;
    const std::vector<T>&           args() const final;
    const size_t&                   Nsys() const final;
    const bool&                     diverges() const final;
    const bool&                     is_running() const final;
    const bool&                     is_dead() const final;
    const vec<T, N>&                error() const final;
    const std::string&              message() const final;
    const SolverState<T, N>         state() const final;
    const EventCollection<T, N>&    events() const final;
    bool                            at_event() const final;
    std::string                     event_name() const final;
    const Event<T, N>&              current_event() const final;
    const int&                      current_event_index() const final;
    const std::string&              name() const final;
    T                               auto_step(T direction=0, const ICS<T, N>* = nullptr) const final;
    OdeSolver<T, N>*                clone() const final;
    UniqueClone                     safe_clone() const final;
    UniqueClone                     with_new_events(const EventCollection<T, N>& events) const final;
    InterpolatorList                interpolators() const final;
    Derived*                        derived_clone() const;

    bool                            advance() final;
    bool                            set_goal(const T& t_max) final;
    void                            stop(const std::string& text) final;
    void                            kill(const std::string& text) final;
    bool                            resume() final;
    bool                            free() final;
    void                            start_interpolation() final;
    void                            stop_interpolation() final;

    void                            clear_interpolators() final;

    inline STATE                    new_state(const T& t, const vec<T, N>& q, const T& h) const;//virtual

    inline void                     adapt_impl(STATE& res, const STATE& state);//virtual

    inline INTERPOLATOR             state_interpolator(const STATE& state1, const STATE& state2, int bdr1, int bdr2) const;//virtual. required only if _req_coef_mat is false

    inline void                     coef_matrix(Eigen::Matrix<T, N, -1>& mat, const STATE& state1, const STATE& state2) const requires _req_coef_mat; //virtual. required only if _req_coef_mat is true



protected:

    DerivedSolver(SOLVER_CONSTRUCTOR(T, N));

    DerivedSolver(const DerivedSolver& other);

    DerivedSolver(DerivedSolver&& other);

    DerivedSolver& operator=(const DerivedSolver& other);

    DerivedSolver& operator=(DerivedSolver&& other);

    const Functor<T, N>& _rhs() const;
    
    const vec<T, N>&     _rhs(const T& t, const vec<T, N>& q) const;

    void                 _rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const;

    const vec<T, N>&     _interp(const T& t) const;

    void                 _finalize(const T& t0, const vec<T, N>& q0, T first_step);

private:

    Event<T, N>& current_event();

    LinkedInterpolator<T, N, INTERPOLATOR>& _current_interpolator();

    const INTERPOLATOR& _interpolator() const;

    void            _register_states();

    void            _initialize_events(const T& t0);

    bool            _adapt_to_event(Event<T, N>& event, const State<T, N>& before, const State<T, N>& after);

    void            _add_interpolant(const INTERPOLATOR& interpolant);

    bool            _validate_it(const STATE& state);

    void            _warn_dead();

    void            _warn_paused();

    void            _warn_travolta();

    void            _find_true_state(const Solver& other);

    void            _finalize_state(const State<T, N>& start);

    void            _clear_states();

    void            _copy_data(const Solver& other);

    void            _move_data(Solver&& other);

    Functor<T, N>                               _ode_rhs;
    T                                           _rtol;
    T                                           _atol;
    T                                           _min_step;
    T                                           _max_step;
    std::vector<T>                              _args;
    size_t                                      _n; //size of ode system
    std::string                                 _name;
    T                                           _tmax;
    bool                                        _diverges = false;
    bool                                        _is_running = true;
    bool                                        _is_dead = false;
    size_t                                      _N=0;//total number of solution updates
    vec<T, N>                                   _error;
    std::string                                 _message;
    EventCollection<T, N>                       _events = {};
    int                                         _current_event_index = -1;
    int                                         _direction;
    STATE*                                      _initial_state;
    STATE*                                      _state;
    STATE*                                      _old_state;
    STATE*                                      _aux_state; //only helps transitioning between naturally adapted states.
    ViewState<T, N>*                            _temp_state; //mostly in case when tmax is met
    const State<T, N>*                          _true_state;
    bool                                        _equiv_states = true; //when true, it is time to adapt the solver to its next natural step (_true_state->t() is equal to _state->t())
    bool                                        _requires_new_start = false; //if true, then _state must restart from _true_state.
    bool                                        _interp_data = false;
    std::vector<LinkedInterpolator<T, N, INTERPOLATOR>>   _interpolators = {};
    mutable _MutableData<T, N, INTERPOLATOR>    _mut;
};



/*
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
------------------------------------------------IMPLEMENTATIONS-------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
*/





template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::DerivedSolver(SOLVER_CONSTRUCTOR(T, N)): OdeSolver<T, N>(), _ode_rhs(rhs.ode_rhs), _rtol(rtol), _atol(atol), _min_step(min_step), _max_step(max_step), _args(args), _n(q0.size()), _name(name), _error(q0.size()), _events(events), _mut(q0, INTERP_ORDER){
    
    if (_min_step < 0){
        throw std::runtime_error("Minimum stepsize must be a non negative number");
    }
    if (_max_step < _min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    _error.setZero();
}



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::DerivedSolver(const DerivedSolver& other) : OdeSolver<T, N>(other), _ode_rhs(other._ode_rhs), _rtol(other._rtol), _atol(other._atol), _min_step(other._min_step), _max_step(other._max_step), _args(other._args), _n(other._n), _name(other._name), _tmax(other._tmax), _diverges(other._diverges), _is_running(other._is_running), _is_dead(other._is_dead), _N(other._N), _error(other._error), _message(other._message), _events(other._events), _current_event_index(other._current_event_index), _direction(other._direction), _equiv_states(other._equiv_states), _requires_new_start(other._requires_new_start), _interp_data(other._interp_data), _interpolators(other._interpolators), _mut(other._mut){
    _initial_state = new STATE(*other._initial_state);
    _state = new STATE(*other._state);
    _old_state = new STATE(*other._old_state);
    _aux_state = new STATE(*other._aux_state);
    _temp_state = new ViewState(*other._temp_state);
    _find_true_state(other);
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::DerivedSolver(DerivedSolver&& other) : OdeSolver<T, N>(std::move(other)), _ode_rhs(std::move(other._ode_rhs)), _rtol(std::move(other._rtol)), _atol(std::move(other._atol)), _min_step(std::move(other._min_step)), _max_step(std::move(other._max_step)), _args(std::move(other._args)), _n(std::move(other._n)), _name(std::move(other._name)), _tmax(std::move(other._tmax)), _diverges(std::move(other._diverges)), _is_running(std::move(other._is_running)), _is_dead(std::move(other._is_dead)), _N(std::move(other._N)), _error(std::move(other._error)), _message(std::move(other._message)), _events(std::move(other._events)), _current_event_index(std::move(other._current_event_index)), _direction(std::move(other._direction)), _initial_state(other._initial_state), _state(other._state), _old_state(other._old_state), _aux_state(other._aux_state), _temp_state(other._temp_state), _equiv_states(std::move(other._equiv_states)), _requires_new_start(other._requires_new_start), _interp_data(other._interp_data), _interpolators(std::move(other._interpolators)), _mut(std::move(other._mut)){
    _find_true_state(other);
    other._initial_state = nullptr;
    other._state = nullptr;
    other._old_state = nullptr;
    other._aux_state = nullptr;
    other._temp_state = nullptr;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::operator=(const DerivedSolver& other){
    OdeSolver<T, N>::operator=(other);
    _copy_data(other);
    return *this;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::operator=(DerivedSolver&& other){
    OdeSolver<T, N>::operator=(std::move(other));
    _move_data(std::move(other));
    return *this;
}











template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::~DerivedSolver() {
    _clear_states();
}



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline OdeRhs<T, N> DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::ode_rhs() const {
    return {this->_ode_rhs, nullptr};
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::t() const {
    return _true_state->t();
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::q() const {
    return _true_state->exposed_vector();
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::q_true() const {
    return _true_state->vector();
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::stepsize() const {
    return _state->habs();
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::tmax() const {
    return _tmax;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const int& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::direction() const {
    return _direction;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::rtol() const {
    return _rtol;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::atol() const {
    return _atol;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::min_step() const {
    return _min_step;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const T& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::max_step() const {
    return _max_step;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const std::vector<T>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::args() const {
    return _args;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const size_t& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::Nsys() const {
    return _n;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const bool& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::diverges() const {
    return _diverges;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const bool& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::is_running() const {
    return _is_running;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const bool& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::is_dead() const {
    return _is_dead;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::error() const {
    return _error;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const std::string& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::message() const {
    return _message;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const SolverState<T, N> DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::state() const {
    return {_true_state->t(), _true_state->exposed_vector(), _state->habs(), event_name(), _diverges, _is_running, _is_dead, _N, _message};
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const EventCollection<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::events() const {
    return _events;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::at_event() const {
    return _current_event_index != -1;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline std::string DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::event_name() const {
    return at_event() ? current_event().name() : "";
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const Event<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::current_event() const {
    return _events[_current_event_index];
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline Event<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::current_event() {
    return _events[_current_event_index];
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const int& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::current_event_index() const {
    return _current_event_index;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const std::string& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::name() const {
    return _name;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
T DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::auto_step(T direction, const ICS<T, N>* ics)const{
    //returns absolute value of emperically determined first step.
    const int dir = (direction == 0) ? _state->direction() : ( (direction > 0) ? 1 : -1);
    const T& t = (ics == nullptr) ? _state->t() : ics->t;
    const vec<T, N>& q = (ics == nullptr) ? _state->vector() : ics->q;

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
    f1 = _rhs(t+h0*dir, y1);

    d2 = rms_norm(((f1-_dq)/scale).eval()) / h0;
    
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = std::max(T(1)/1000000, h0/1000);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(Derived::ERR_EST_ORDER+1));
    }

    return std::max(std::min({100*h0, h1, this->_max_step}), this->_min_step);
}



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline OdeSolver<T, N>* DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::clone()const{
    return new Derived(*static_cast<const Derived*>(this));
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline std::unique_ptr<OdeSolver<T, N>> DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::safe_clone()const{
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline std::unique_ptr<OdeSolver<T, N>> DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::with_new_events(const EventCollection<T, N>& events)const{
    Derived* cl = derived_clone();
    cl->_events = events;
    cl->_initialize_events(this->t());
    return std::unique_ptr<OdeSolver<T, N>>(cl);
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
std::vector<const Interpolator<T, N>*> DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::interpolators() const{
    std::vector<const Interpolator<T, N>*> res(_interpolators.size());
    for (size_t i=0; i<res.size(); i++){
        res[i] = &_interpolators[i];
    }
    return res;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline Derived* DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::derived_clone()const{
    return new Derived(*static_cast<const Derived*>(this));
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::advance(){

    if (this->_is_dead){
        this->_warn_dead();
        return false;
    }
    else if (!this->_is_running){
        this->_warn_paused();
        return false;
    }

    if (_requires_new_start){
        *_state = std::move(this->new_state(_true_state->t(), _true_state->vector(), (_true_state->h() != 0) ? _true_state->h() : _state->h())); //we could just choose _state->h(), but if the direction has just been changed, _state->h() has the wrong sign, and the correct information is stored in _true_state->h(). Always prefer this unless it is zero (unspecified).
        _requires_new_start = false;
        _equiv_states = true;
    }
    
    if (_equiv_states){
        this->adapt_impl(*_aux_state, *_state); //only *_aux_state changed
        if (_validate_it(*_aux_state)){
            _register_states(); //now _old_state pointer took over the _state pointer, _state is updated, and _true_state points to _state
            _finalize_state(*_old_state); //mainly affects the _true_state pointer, unless a masked event is encountered
            this->_error += _true_state->local_error();
        }
        else{
            return false;
        }
    }
    else{
        //we need to advance from _true_state to _state.
        //We might encounter an event between these two.
        const State<T, N>* tmp = _true_state;
        _true_state = _state; //temporarily set to the next naturally adapted state.
        _equiv_states = true;
        _finalize_state(*tmp);
    }
    return true;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::set_goal(const T& t_max_new){
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
        this->_tmax = t_max_new;
        const int dir = sgn(t_max_new-_true_state->t());
        if (dir*_state->direction() < 0){
            _direction = dir;
            ICS<T, N> ics{_true_state->t(), _true_state->vector()};
            *_temp_state = std::move(ViewState<T, N>(ics.t, ics.q, _true_state->exposed_vector(), auto_step(dir, &ics)));
            _true_state = _temp_state;
            _requires_new_start = true;
            if (_interp_data){
                stop_interpolation();
                start_interpolation();
            }
        }
        return resume();
    }
}



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::stop(const std::string& text){
    _is_running = false;
    _message = (text == "") ? "Stopped by user" : text;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::kill(const std::string& text){
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::resume(){
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


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::free() {
    if (_direction < 0){
        return set_goal(-inf<T>());
    }
    else{
        //default direction is positive
        return set_goal(inf<T>());
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::start_interpolation() {
    if (!_interp_data){
        _interp_data = true;
        _interpolators.push_back(LinkedInterpolator<T, N, INTERPOLATOR>(_true_state->t(), _true_state->exposed_vector()));
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::stop_interpolation() {
    _interp_data = false;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::clear_interpolators() {
    _interpolators.clear();
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
const INTERPOLATOR& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_interpolator() const{
    if (_mut.interpolator_is_set){
        return _mut.interpolator;
    }
    else{
        _mut.interpolator = std::move(state_interpolator(*_old_state, *_state, 0, 0));
        _mut.interpolator.adjust(_true_state->t());
        _mut.interpolator_is_set = true;
        return _mut.interpolator;
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline STATE DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::new_state(const T& t, const vec<T, N>& q, const T& h) const {
    return static_cast<const Derived*>(this)->new_state(t, q, h);
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::adapt_impl(STATE& res, const STATE& state){
    return static_cast<Derived*>(this)->adapt_impl(res, state);
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline INTERPOLATOR DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::state_interpolator(const STATE& state1, const STATE& state2, int bdr1, int bdr2) const {
    if constexpr (_req_coef_mat){
        coef_matrix(_mut.coef_mat, state1, state2);
        return INTERPOLATOR(_mut.coef_mat, state1, state2, bdr1, bdr2);
    }
    else{
        return static_cast<const Derived*>(this)->state_interpolator(state1, state2, bdr1, bdr2);
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::coef_matrix(Eigen::Matrix<T, N, -1>& mat, const STATE& state1, const STATE& state2) const requires _req_coef_mat{
    return static_cast<const Derived*>(this)->coef_matrix(mat, state1, state2);
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const Functor<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_rhs() const{
    return _ode_rhs;
}



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_rhs(vec<T, N>& result, const T& t, const vec<T, N>& q) const{
    _ode_rhs(result, t, q, _args);
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline const vec<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_rhs(const T& t, const vec<T, N>& q) const {
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


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
const vec<T, N>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_interp(const T& t) const{
    this->_interpolator().call(_mut.q, t);
    return _mut.q;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_finalize(const T& t0, const vec<T, N>& q0, T first_step){

    if (!q0.isFinite().all()){
        throw std::runtime_error("The given initial conditions contain non finite values (inf or nan).");
    }

    int dir = sgn(first_step);

    if (first_step != 0){
        first_step = choose_step(abs(first_step), _min_step, _max_step);
    }
    else{
        const ICS<T, N> ics = {t0, q0};
        first_step = this->auto_step(1, &ics);
        dir = 1;
    }

    _initialize_events(t0);
    
    _direction = dir;
    //now first_step and initial direction are both != 0.
    _initial_state = new STATE(new_state(t0, q0, first_step*dir));
    _state = new STATE(*_initial_state);
    _old_state = new STATE(*_initial_state);
    _aux_state = new STATE(*_initial_state);
    _temp_state = new ViewState<T, N>(t0, q0);
    _true_state = _state;
}




template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>& DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_current_interpolator(){
    return _interpolators.back();
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_register_states(){
    STATE* tmp = _state;
    _state = _aux_state;
    _aux_state = _old_state;
    _old_state = tmp;
    _true_state = _state; //default value of _true_state, but it might change before the step is finalized
    _equiv_states = true;
    _mut.interpolator_is_set = false;
    if (_interp_data){
        this->_add_interpolant(state_interpolator(*_old_state, *_state, 0, -1));
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_initialize_events(const T& t0){

    _events.set_args(this->_args);
    for (size_t i=0; i<_events.size(); i++){
        if (PeriodicEvent<T, N>* p = dynamic_cast<PeriodicEvent<T, N>*>(&_events[i])){
            if (abs(p->t_start()) == inf<T>()){
                p->set_start(t0+p->period());
            }
            else if (p->t_start() == t0){
                throw std::runtime_error("The starting time of a periodic event cannot be set at the initial time of the ode solver.");
            }
        }
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_adapt_to_event(Event<T, N>& event, const State<T, N>& before, const State<T, N>& after){

    //MUST NOT change _state or _old_state in here, only _true_state

    std::function<vec<T, N>(T)> qfunc = [this](const T& t){
        return this->_interp(t);
    };
    
    if (event.determine(before.t(), before.vector(), after.t(), after.vector(), qfunc)){
        _true_state = &event.state();
        _equiv_states = false;
        return true;
    }
    return false;
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_add_interpolant(const INTERPOLATOR& interpolant){
    LinkedInterpolator<T, N, INTERPOLATOR>& r = _current_interpolator();
    if (r.last_interpolant().interval().is_point() && interpolant.interval().start_bdr() == 0){
        INTERPOLATOR tmp(interpolant);
        tmp.close_start();
        r.expand(tmp);
    }
    else{
        r.expand(interpolant);
    }
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
bool DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_validate_it(const STATE& state){
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

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_warn_dead() {
    std::cout << std::endl << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << "\n";
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_warn_paused() {
    std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
}

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_warn_travolta() {
    std::cout << std::endl << "Solver has not been specified an integration direction, possibly because the Tmax goal was reached. Please set a new Tmax goal first or free() the solver.\n";
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_find_true_state(const Solver& other){
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


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_finalize_state(const State<T, N>& start){

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

    if (_true_state->t()*_direction >= this->_tmax*_direction){

        if (_true_state->t()==_tmax){}
        else if (this->at_event()){
            //sometimes an event might appear a bit ahead of the tmax. This has already been registered
            //so we need to un-register it before stopping. It will be encoutered anyway when the solver is resumed.

            this->current_event().go_back();
            this->_current_event_index = -1;
            //Since the tmax is before the masked event, the interpolating coefficients can still be used
            //even if they range from _old_state to _state, where _state appears *after* the event.
            *_temp_state = std::move(ViewState<T, N>(_tmax, _interp(_tmax)));
            _true_state = _temp_state;
            _equiv_states = false;
        }
        else{
            *_temp_state = std::move(ViewState<T, N>(_tmax, _interp(_tmax)));
            _true_state = _temp_state;
            _equiv_states = false;
        }

        this->stop("T_max goal reached");
    }
    else if (this->at_event()){

        if (this->current_event().is_masked()){
            _requires_new_start = true;
            _mut.interpolator_is_set = false;
            
            if (_interp_data){
                if (this->current_event().hides_mask()){
                    _current_interpolator().close_end();
                }
                else{
                    _add_interpolant(INTERPOLATOR(_true_state->t(), _true_state->exposed_vector()));
                }
            }
        }
        else if(!this->current_event().is_precise()){
            _equiv_states = true;
        }

        if (this->current_event().is_leathal()){
            this->kill(this->current_event().name());
        }
        else if (this->current_event().is_stop_event()){
            this->stop(this->current_event().name());
        }
    }

    if (_interp_data){
        _current_interpolator().adjust(_true_state->t());
    }


    this->_N++;
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_clear_states(){
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


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_copy_data(const Solver& other){
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
    _events = other._events;
    _current_event_index = other._current_event_index;
    _direction = other._direction;
    _equiv_states = other._equiv_states;
    _requires_new_start = other._requires_new_start;
    _interp_data = other._interp_data;
    _interpolators = other._interpolators;
    _mut = other._mut;

    *_initial_state = *other._initial_state;
    *_state = *other._state;
    *_old_state = *other._old_state;
    *_aux_state = *other._aux_state;
    *_temp_state = *other._temp_state;
    _find_true_state(other);
}


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
inline void DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::_move_data(Solver&& other) {
    _ode_rhs = std::move(other._ode_rhs);
    _rtol = std::move(other._rtol);
    _atol = std::move(other._atol);
    _min_step = std::move(other._min_step);
    _max_step = std::move(other._max_step);
    _args = std::move(other._args);
    _n = std::move(other._n);
    _name = std::move(other._name);
    _tmax = std::move(other._tmax);
    _diverges = std::move(other._diverges);
    _is_running = std::move(other._is_running);
    _is_dead = std::move(other._is_dead);
    _N = std::move(other._N);
    _error = std::move(other._error);
    _message = std::move(other._message);
    _events = std::move(other._events);
    _current_event_index = std::move(other._current_event_index);
    _direction = std::move(other._direction);
    _equiv_states = std::move(other._equiv_states);
    _requires_new_start = other._requires_new_start;
    _interp_data = other._interp_data;
    _interpolators = std::move(other._interpolators);
    _mut = std::move(other._mut);

    //We could move the pointers themselves which is faster (the 5 objects below are dynamically allocated),
    //but this below is safer. The _true_state pointer will not need to change, since no
    //addresses have changed.
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


template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
const T DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::MAX_FACTOR = T(10);

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
const T DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::SAFETY = T(9)/10;

template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
const T DerivedSolver<T, N, Derived, STATE, INTERPOLATOR>::MIN_FACTOR = T(2)/10;


#endif