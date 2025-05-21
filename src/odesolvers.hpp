#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "events.hpp"
#include <limits>



template<class T, int N>
struct SolverArgs{

    const Jac<T, N> jac;
    const T t0;
    const vec<T, N> q0;
    const T rtol;
    const T atol;
    const T h_min;
    const T h_max;
    const T first_step;
    const std::vector<T> args;
    const std::vector<Event<T, N>*> events;
    const Func<T, N> mask;
    const std::string save_dir;
    const bool save_events_only;
};

template<class T, int N>
struct _MutableData{

    _MutableData(const vec<T, N>& q) : q(q), jac(q), state({0, q, 0}){}

    _MutableData(){}

    vec<T, N> q;
    vec<T, N> jac;
    State<T, N> state;
};


template<class T, int N>
void write_checkpoint(std::ofstream& file, const T& t, const vec<T, N>& q, const int& event_index);


template<class T, int N>
class OdeSolver{


public:

    const T MAX_FACTOR = T(10);
    const T SAFETY = T(9)/10;
    const T MIN_FACTOR = T(2)/10;
    const int ORDER;
    const int ERR_EST_ORDER;


    virtual ~OdeSolver();

    //MODIFIERS
    void stop(const std::string& text = "");
    void kill(const std::string& text = "");
    bool advance_by(const T& habs);
    bool advance_by_any(const T& h);
    bool advance();
    bool set_goal(const T& t_max);

    //ACCESSORS
    inline const T& t() const;
    inline const vec<T, N>& q() const;
    inline const vec<T, N>& q_true() const;
    inline const T& stepsize() const;
    inline const T& tmax() const;
    inline const int& direction() const;
    inline const T& rtol() const;
    inline const T& atol() const;
    inline const T& h_min() const;
    inline const T& h_max() const;
    inline const std::vector<T>& args() const;
    inline const size_t& Nsys() const;
    inline const bool& diverges() const;
    inline const bool& is_stiff() const;
    inline const bool& is_running() const;
    inline const bool& is_dead() const;
    inline const std::string& message();
    inline const bool& autosave() const;
    bool resume();
    bool release_file();
    inline bool file_is_ready() const;
    bool reopen_file();
    inline const std::string& filename() const;
    bool free();
    inline const bool at_event() const;
    inline std::string event_name() const;
    inline const SolverState<T, N> state() const;
    inline const Event<T, N>* current_event() const;
    inline const int& current_event_index() const;
    inline const Event<T, N>* event(const size_t& i);
    inline size_t events_size() const;


    virtual void apply_step(vec<T, N>& q_new, const T& t_old, const vec<T, N>& q_old, const T& h) const = 0;

    virtual State<T, N> adaptive_step() const = 0; //derived implementation must account for h_min

    virtual OdeSolver<T, N>* clone() const = 0;

    T auto_step()const;

protected:

    OdeSolver(const SolverArgs<T, N>& S, const int& order, const int& err_est_order);

    OdeSolver(const OdeSolver<T, N>& other);

    OdeSolver(OdeSolver<T, N>&& other);

    OdeSolver<T, N>& operator=(const OdeSolver<T, N>& other);

    const vec<T, N>& _jac(const T& t, const vec<T, N>& q) const;

    const vec<T, N>& _step(const T& t_old, const vec<T, N>& q_old, const T& h) const;



private:

    Jac<T, N> _fill_jac;
    T _t;
    vec<T, N> _q;
    T _rtol;
    T _atol;
    T _h_min;
    T _h_max;
    T _habs;
    std::vector<T> _args;

    size_t _n; //size of ode system
    T _tmax;
    bool _diverges = false;
    bool _is_stiff = false;
    bool _is_running = true;
    bool _is_dead = false;
    size_t _N=0;//total number of solution updates
    std::string _message; //different from "running".
    int _direction;
    std::vector<Event<T, N>*> _events;
    int _current_event_index = -1;
    std::string _filename;
    std::ofstream _file;
    bool _autosave = false;
    bool _save_events_only = false;
    State<T, N>* _checkpoint = nullptr;
    const vec<T, N>* _q_exposed = nullptr; //view_only pointer
    mutable _MutableData<T, N> _mut;

    bool _adapt_to_event(State<T, N>& next, Event<T, N>& event);

    bool _go_to_state();

    bool _update(const T& t_new, const vec<T, N>& y_new, const T& h_next);

    void _warn_dead();

    void _warn_paused();

    void _warn_travolta();

    void _copy_data(const OdeSolver<T, N>& other);

    void _make_new_events(const std::vector<Event<T, N>*>& events);

    void _delete_events();

};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/

template<class T, int N>
OdeSolver<T, N>::~OdeSolver(){
    delete _checkpoint;
    _checkpoint = nullptr;
    _delete_events();
    if (_autosave){
        _file.close();
    }
};


template<class T, int N>
inline const T& OdeSolver<T, N>::t() const { return _t; }

template<class T, int N>
inline const vec<T, N>& OdeSolver<T, N>::q() const { return *_q_exposed; }

template<class T, int N>
inline const vec<T, N>& OdeSolver<T, N>::q_true() const { return _q; }

template<class T, int N>
inline const T& OdeSolver<T, N>::stepsize() const { return _habs; }

template<class T, int N>
inline const T& OdeSolver<T, N>::tmax() const { return _tmax; }

template<class T, int N>
inline const int& OdeSolver<T, N>::direction() const { return _direction; }

template<class T, int N>
inline const T& OdeSolver<T, N>::rtol() const { return _rtol; }

template<class T, int N>
inline const T& OdeSolver<T, N>::atol() const { return _atol; }

template<class T, int N>
inline const T& OdeSolver<T, N>::h_min() const { return _h_min; }

template<class T, int N>
inline const T& OdeSolver<T, N>::h_max() const { return _h_max; }

template<class T, int N>
inline const std::vector<T>& OdeSolver<T, N>::args() const { return _args; }

template<class T, int N>
inline const size_t& OdeSolver<T, N>::Nsys() const { return _n; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::diverges() const { return _diverges; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::is_stiff() const { return _is_stiff; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::is_running() const { return _is_running; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::is_dead() const { return _is_dead; }

template<class T, int N>
inline const std::string& OdeSolver<T, N>::message() { return _message; }

template<class T, int N>
inline const bool& OdeSolver<T, N>::autosave() const { return _autosave; }

template<class T, int N>
T OdeSolver<T, N>::auto_step()const{
    //returns absolute value of emperically determined first step.
    if (_direction == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    T h0, d2, h1;
    vec<T, N> y1, f1;
    vec<T, N> scale = _atol + cwise_abs(_q)*_rtol;
    vec<T, N> _dq = this->_jac(_t, _q);
    T d0 = rms_norm((_q/scale).eval());
    T d1 = rms_norm((_dq/scale).eval());
    if (d0 < 1e-5 || d1 < 1e-5){
        h0 = 1e-6;
    }
    else{
        h0 = 0.01*d0/d1;
    }
    y1 = _q+h0*_direction*_dq;
    f1 = _jac(_t+h0*_direction, y1);
    d2 = rms_norm(((f1-_dq)/scale).eval()) / h0;
    
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = std::max(T(1e-6), 1e-3*h0);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }

    return std::min({100*h0, h1, _h_max});
}

template<class T, int N>
bool OdeSolver<T, N>::resume() {
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
bool OdeSolver<T, N>::free() {
    if (_direction == -1){
        return set_goal(-inf<T>());
    }
    else{
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
    return {_t, q(), _habs, event_name(), _diverges, _is_stiff, _is_running, _is_dead, _N, _message};
}

template<class T, int N>
inline const Event<T, N>* OdeSolver<T, N>::current_event() const {
    return at_event() ? _events[_current_event_index] : nullptr;
}

template<class T, int N>
inline const int& OdeSolver<T, N>::current_event_index() const {
    return _current_event_index;
}

template<class T, int N>
inline const Event<T, N>* OdeSolver<T, N>::event(const size_t& i) {
    return _events[i];
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
    if ((_diverges) && (!_is_dead || _is_running) ){
        //sanity check. 
        throw std::runtime_error("Bug detected: Solver half alive");
    }

    if (_is_dead){
        _warn_dead();
        return false;
    }
    else if (t_max_new == _t){
        _direction = 0;
        _tmax = t_max_new;
        stop("Waiting for new Tmax");
        return true;
    }
    else{
        _tmax = t_max_new;
        _direction = ( t_max_new > _t) ? 1 : -1;
        if (_habs == 0){
            _habs = auto_step();
        }
        return resume();
    }
}







template<typename T, int N>
OdeSolver<T, N>::OdeSolver(const SolverArgs<T, N>& S, const int& order, const int& err_est_order): ORDER(order), ERR_EST_ORDER(err_est_order), _fill_jac(S.jac), _t(S.t0), _q(S.q0), _rtol(S.rtol), _atol(S.atol), _h_min(S.h_min), _h_max(S.h_max), _args(S.args), _n(S.q0.size()), _filename(S.save_dir), _save_events_only(S.save_events_only), _mut(S.q0){
    if (_h_max < _h_min){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    if (S.first_step <= 0){
        _habs = 0;
    }
    else{
        _habs = (S.first_step < _h_min) ? _h_min : S.first_step;
    }

    if (S.mask != nullptr){
        Jac<T, N> f_tmp = S.jac;
        Func<T, N> msk = S.mask;
        _fill_jac = [msk, f_tmp](vec<T, N>& dq, const T& t, const vec<T, N>& q, const std::vector<T>& args) {
            f_tmp(dq, t, q, args);
            dq = msk(t, dq, args);
        };
    }

    _q_exposed = &_q;
    _make_new_events(S.events);
    set_goal(_t);

    if (!_filename.empty()){
        if (typeid(T) != typeid(_q[0])){
            throw std::runtime_error("Cannot turn on autosaving to OdeSolver whose step is not 1D");
        }
        _file.open(_filename, std::ios::out);
        if (!_file){
            throw std::runtime_error("Could not open file in OdeSolver for automatic saving: " + _filename + "\n");
        }
        _autosave = true;
        write_checkpoint(_file, _t, _q, -1);
    }
}

template<typename T, int N>
OdeSolver<T, N>::OdeSolver(const OdeSolver<T, N>& other): ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER){
    _copy_data(other);
    _filename = "";
    _autosave = false;
}

template<typename T, int N>
OdeSolver<T, N>::OdeSolver(OdeSolver<T, N>&& other): ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER), _file(std::move(other._file)){
    // not the most efficient, but the most readable :)
    // besides the time scale of a copy is insignificant to the timescale of
    // solving an ode.
    _copy_data(other);
}

template<typename T, int N>
OdeSolver<T, N>& OdeSolver<T, N>::operator=(const OdeSolver<T, N>& other){
    _copy_data(other);
    _filename = "";
    _autosave = false;
    return *this;
}

template<typename T, int N>
inline const vec<T, N>& OdeSolver<T, N>::_jac(const T& t, const vec<T, N>& q) const {
    /*
    only assign a copy:
        e.g. vec<T, N> r = this->_jac(...); allowed

        do NOT do this:
            const vec<T, N>& r = this->_jac(...);

    This is because a reference to a mutable object is returned
    to improve efficiency. If the function is called with different arguments,
    the underlying mutable object will change value, which will lead to unexpected behavior
    when using the first const& defined.
    */
    _fill_jac(_mut.jac, t, q, _args);
    return _mut.jac;
}

template<typename T, int N>
const vec<T, N>& OdeSolver<T, N>::_step(const T& t_old, const vec<T, N>& q_old, const T& h) const{
    //also use only to assign directly
    apply_step(_mut.q, t_old, q_old, h);
    return _mut.q;
}









template<class T, int N>
bool OdeSolver<T, N>::advance(){
    _mut.state = adaptive_step();
    return _go_to_state();
}



template<class T, int N>
bool OdeSolver<T, N>::advance_by(const T& habs){
    if (habs <= 0){
        std::cout << std::endl << "Please provide a positive stepsize in .advance_by(habs)\n";
        return false;
    }

    delete _checkpoint;
    _checkpoint = nullptr;

    bool _set_non_stiff = false;
    if (habs <= _h_min && !_is_stiff){
        _set_non_stiff = true;
    }
    State<T, N>& state = _mut.state;
    state.h_next = habs;
    state.t = _t+habs*_direction;
    state.q = this->_step(_t, _q, _habs*_direction);
    bool success = _go_to_state();
    if (success && _set_non_stiff){
        _is_stiff = false;
    }
    return success;
}

template<class T, int N>
bool OdeSolver<T, N>::advance_by_any(const T& h){
    delete _checkpoint;
    _checkpoint = nullptr;

    set_goal(_t+h);
    State<T, N>& state = _mut.state;
    state.t = _t+h;
    state.h_next = h*_direction;
    state.q = this->_step(_t, _q, h);
    return _go_to_state();
}

template<class T, int N>
bool OdeSolver<T, N>::_update(const T& t_new, const vec<T, N>& y_new, const T& h_next){
    
    bool success = true;
    if (h_next < 0){//h_next is always positive, it is the absolute value of the true stepsize
        success = false;
        throw std::runtime_error("Bug detected: Absolute stepsize < 0");
    }

    if (!y_new.isFinite().all()){
        kill("Ode solution diverges");
        _diverges = true;
        success = false;
    }
    else if (h_next == 0){
        kill("Required stepsize was smaller than machine precision");
        _is_stiff = true;
        success = false;
    }

    //make or clear checkpoint first
    if (_current_event_index == -1 && _checkpoint != nullptr){
        delete _checkpoint;
        _checkpoint = nullptr;
    }


    if (t_new*_direction >= _tmax*_direction){
        if (t_new == _tmax){
            _q = y_new;
        }
        else if (_current_event_index != -1){
            //sometimes an event might appear a bit ahead of the tmax. This has already been registered
            //so we need to un-register it before stopping. It will be encoutered anyway when the solver is resumed.
            _events[_current_event_index]->go_back();
            _current_event_index = -1;
            _q = this->_step(_t, _q, _tmax-_t);
        }
        else{
            _q = this->_step(_t, _q, _tmax-_t);
        }
        stop("T_max goal reached");
        _t = _tmax;
        _habs = h_next;
        _N++;
    }
    else{
        if ( h_next == _h_min){
            _is_stiff = true;
            if (h_next > _habs){
                throw std::runtime_error("Bug detected in min_step implementation (1).");
            }
        }
        else if (h_next < _h_min){
            throw std::runtime_error("Bug detected in min_step implementation (2).");
        }
        _t = t_new;
        _q = y_new;
        _habs = h_next;
        _N++;

    }

    if (_autosave && success){
        if (!_save_events_only || (_current_event_index != -1)){
            write_checkpoint(_file, _t, q(), _current_event_index);
        }
    }

    if (at_event()){
        if (current_event()->is_leathal()){
            kill(current_event()->name());
            success = false;
        }
        else if (current_event()->is_stop_event()){
            stop(current_event()->name());
        }
    }

    return success;
}

template<class T, int N>
bool OdeSolver<T, N>::_adapt_to_event(State<T, N>& next, Event<T, N>& event){
    // takes next state (which means tnew, hnew, and hnext_new)
    // if it is not an event or smth it is left unchanged.
    // otherwise, it is modified to depict the event with high accuracy
    std::function<vec<T, N>(T)> qfunc = [this](const T& t_next) -> vec<T, N> {
        return this->_step(this->_t, this->_q, t_next-this->_t);
    };
    
    if (event.determine(this->_t, this->_q, next.t, next.q, qfunc)){
        if (_current_event_index == -1 && event.allows_checkpoint()){
            _checkpoint = new State<T, N>{next.t, next.q, next.h_next};
        }
        else if ( _checkpoint != nullptr && event.has_mask()){
            delete _checkpoint;
            _checkpoint = nullptr;
        }
        next.t = event.data().t();
        next.q = event.data().q_true();
        if (event.hide_mask()){
            _q_exposed = &event.data().q();
        }
        return true;
    }
    return false;
}

template<class T, int N>
bool OdeSolver<T, N>::_go_to_state(){

    if (_is_dead){
        _warn_dead();
        return false;
    }
    else if (!_is_running){
        _warn_paused();
        return false;
    }
    else {
        _current_event_index = -1;
        _q_exposed = &_q;
        State<T, N>& next = _mut.state;
        if (_checkpoint != nullptr){
            next = *_checkpoint;
        }

        for (int i=0; i<static_cast<int>(_events.size()); i++){
            if (_adapt_to_event(next, *_events[i])){
                if (_current_event_index != -1){
                    _events[_current_event_index]->go_back();
                }
                _current_event_index = i;
                if (!current_event()->is_precise()){
                    break;
                }
            }
        }

        return _update(next.t, next.q, next.h_next);
    }

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
    _fill_jac = other._fill_jac;
    _t = other._t;
    _q = other._q;
    _rtol = other._rtol;
    _atol = other._atol;
    _h_min = other._h_min;
    _h_max = other._h_max;
    _habs = other._habs;
    _args = other._args;
    _n = other._n;
    _tmax = other._tmax;
    _diverges = other._diverges;
    _is_stiff = other._is_stiff;
    _is_running = other._is_running;
    _is_dead = other._is_dead;
    _N = other._N;
    _message = other._message;
    _direction = other._direction;
    _current_event_index = other._current_event_index;
    _save_events_only = other._save_events_only;

    _filename = other._filename;
    _autosave = other._autosave;

    _make_new_events(other._events);
    delete _checkpoint;
    _checkpoint = nullptr;
    if (other._checkpoint != nullptr){
       _checkpoint = new State<T, N>(*other._checkpoint);
    }
    if (other._q_exposed == &other._q){
        _q_exposed = &_q;
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



template<class T, int N>
void write_checkpoint(std::ofstream& file, const T& t, const vec<T, N>& q, const int& event_index){
    file << event_index << " " << std::setprecision(16) << t;
    for (size_t i=0; i<static_cast<size_t>(q.size()); i++){
        file << " " << std::setprecision(16) << q[i];
    }
    file << "\n";
}


#endif



