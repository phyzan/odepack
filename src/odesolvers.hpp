#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "events.hpp"
#include <limits>



template<class T, int N>
struct SolverArgs{

    const Func<T, N> f;
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
void write_chechpoint(std::ofstream& file, const T& t, const vec<T, N>& q, const int& event_index){
    file << event_index << " " << std::setprecision(16) << t;
    for (size_t i=0; i<static_cast<size_t>(q.size()); i++){
        file << " " << std::setprecision(16) << q[i];
    }
    file << "\n";
}


template<class T, int N>
class OdeSolver{


public:

    using Callable = Func<T, N>;
    const T MAX_FACTOR = T(10);
    const T SAFETY = T(9)/10;
    const T MIN_FACTOR = T(2)/10;
    const int ORDER;
    const int ERR_EST_ORDER;


    virtual ~OdeSolver(){
        _clear_checkpoint();
        _delete_events();
        if (_autosave){
            _file.close();
        }
    };

    //MODIFIERS
    void stop(const std::string& text = "") {_is_running = false; _message = (text == "") ? "Stopped by user" : text;}
    void kill(const std::string& text = "") {_is_running = false; _is_dead = true; _message = (text == "") ? "Killed by user" : text;}
    bool advance_by(const T& habs);
    bool advance_by_any(const T& h);
    bool advance();
    bool set_goal(const T& t_max);

    //ACCESSORS
    const T& t() const { return _t; }
    const vec<T, N>& q() const { return *_q_exposed; }
    const vec<T, N>& q_true() const { return _q; }

    const T& stepsize() const { return _habs; }
    const T& tmax() const { return _tmax; }
    const int& direction() const { return _direction; }
    const T& rtol() const { return _rtol; }
    const T& atol() const { return _atol; }
    const T& h_min() const { return _h_min; }
    const T& h_max() const { return _h_max; }
    const std::vector<T>& args() const { return _args; }
    const size_t& Nsys() const { return _n; }
    const bool& diverges() const {return _diverges;}
    const bool& is_stiff() const {return _is_stiff;}
    const bool& is_running() const {return _is_running;}
    const bool& is_dead() const {return _is_dead;}
    const std::string& message() {return _message;}
    const bool& autosave() const{
        return _autosave;
    }

    bool resume(){
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

    bool release_file(){
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

    bool file_is_ready()const{
        return _file.good();
    }

    bool reopen_file(){
        if (_autosave || _filename.empty()){
            return false;
        }
        else{
            _file.open(_filename, std::ios::app);
            _autosave = true;
            return true;
        }
    }

    const std::string& filename()const{
        return _filename;
    }

    bool free(){
        if (_direction == -1){
            return set_goal(-inf<T>());
        }
        else{
            return set_goal(inf<T>());
        }

    }

    const bool at_event()const{
        return _current_event_index != -1;
    }

    std::string event_name() const{
        return at_event() ? current_event()->name() : "";
    }

    const SolverState<T, N> state() const {
        return {_t, q(), _habs, event_name(), _diverges, _is_stiff, _is_running, _is_dead, _N, _message};
    }

    const Event<T, N>* current_event() const{
        //we need pointer and not reference, because it might be null
        return at_event() ? _events[_current_event_index] : nullptr;
    }

    const int& current_event_index() const{
        //we need pointer and not reference, because it might be null
        return _current_event_index;
    }

    const Event<T, N>* event(const size_t& i){
        return _events[i];
    }

    size_t events_size()const{
        return _events.size();
    }


    //MEMBER FUNCTIONS BELOW IMPLEMENTED BY CUSTOM DERIVED CLASSES
    //THEY FIRST 2 MUST NOT DEPEND ON THE CURRENT STATE

    virtual vec<T, N> step(const T& t_old, const vec<T, N>& q_old, const T& h) const = 0;

    virtual State<T, N> adaptive_step() const = 0; //derived implementation must account for h_min

    virtual OdeSolver<T, N>* clone() const = 0;

    T auto_step()const{
        //returns absolute value of emperically determined first step.
        if (_direction == 0){
            //needed even if the resulting stepsize will have a positive value.
            throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
        }
        T h0, d2, h1;
        vec<T, N> y1, f1;
        vec<T, N> scale = _atol + cwise_abs(_q)*_rtol;
        vec<T, N> _dq = _f(_t, _q, _args);
        T d0 = rms_norm((_q/scale).eval());
        T d1 = rms_norm((_dq/scale).eval());
        if (d0 < 1e-5 || d1 < 1e-5){
            h0 = 1e-6;
        }
        else{
            h0 = 0.01*d0/d1;
        }
        y1 = _q+h0*_direction*_dq;
        f1 = _f(_t+h0*_direction, y1, _args);
        d2 = rms_norm(((f1-_dq)/scale).eval()) / h0;
        
        if (d1 <= 1e-15 && d2 <= 1e-15){
            h1 = std::max(T(1e-6), 1e-3*h0);
        }
        else{
            h1 = pow(100*std::max(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
        }

        return std::min({100*h0, h1, _h_max});
    }



protected:

    OdeSolver(const SolverArgs<T, N>& S, const int& order, const int& err_est_order): ORDER(order), ERR_EST_ORDER(err_est_order), _f(S.f), _t(S.t0), _q(S.q0), _rtol(S.rtol), _atol(S.atol), _h_min(S.h_min), _h_max(S.h_max), _args(S.args), _n(S.q0.size()), _filename(S.save_dir), _save_events_only(S.save_events_only){
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
            Func<T, N> f_tmp = S.f;
            Func<T, N> msk = S.mask;
            _f = [msk, f_tmp](const T& t, const vec<T, N>& q, const std::vector<T>& args)->vec<T, N> {
                return msk(t, f_tmp(t, q, args), args);
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
            write_chechpoint(_file, _t, _q, -1);
        }
    }

    OdeSolver(const OdeSolver<T, N>& other) : ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER){
        _copy_data(other);
        _filename = "";
        _autosave = false;
    };

    OdeSolver(OdeSolver<T, N>&& other): ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER), _file(std::move(other._file)){
        //not the most efficient, but the most readable :)
        //besides the time scale of a copy is insignificant to the timescale of
        //solving an ode.
        _copy_data(other);
    }


    OdeSolver<T, N>& operator=(const OdeSolver<T, N>& other){
        _copy_data(other);
        _filename = "";
        _autosave = false;
        return *this;
    }

    inline vec<T, N> f(const T& t, const vec<T, N>& q)const{
        return _f(t, q, _args);
    }



private:

    Callable _f;
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
    T* _t_check = nullptr;
    vec<T, N>* _q_check = nullptr;
    T* _habs_check = nullptr;
    const vec<T, N>* _q_exposed = nullptr; //view_only pointer

    vec<T, N> _q_step(const T& t_next)const{
        return this->step(this->_t, this->_q, t_next-this->_t);
    }

    bool _adapt_to_event(State<T, N>& next, Event<T, N>& event);

    bool _go_to_state(State<T, N>& next);

    bool _update(const T& t_new, const vec<T, N>& y_new, const T& h_next);

    void _warn_dead(){
        std::cout << std::endl << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << "\n";
    }

    void _warn_paused(){
        std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
    }

    void _warn_travolta(){
        std::cout << std::endl << "Solver has not been specified an integration direction, possibly because the Tmax goal was reached. Please set a new Tmax goal first or free() the solver.\n";
    }

    void _copy_data(const OdeSolver<T, N>& other){
        //does not copy _file, this has to be managed outside this function
        _f = other._f;
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
        _clear_checkpoint();
        if (other._t_check != nullptr){
            _make_checkpoint(*other._t_check, *other._q_check, *other._habs_check);
        }
        if (other._q_exposed == &other._q){
            _q_exposed = &_q;
        }
        else{
            _q_exposed = &current_event()->data().q();
        }
    }

    void _make_checkpoint(const T& t, const vec<T, N>& q, const T& habs){
        _t_check = new T;
        _q_check = new vec<T, N>;
        _habs_check = new T;
        *_t_check = t;
        *_q_check = q;
        *_habs_check = habs;
    }

    void _clear_checkpoint(){
        delete _t_check;
        delete _q_check;
        delete _habs_check;
        _t_check = nullptr;
        _q_check = nullptr;
        _habs_check = nullptr;
    }

    void _make_new_events(const std::vector<Event<T, N>*>& events){

        //FIRST create a new vector with new allocated objects, because "events" might be
        //our current _events vector. We sort the vector to contain normal events first,
        //and stop_events after to improve runtime performance and not miss out on any stop_events
        //if a single step encouters multiple events.
        std::vector<Event<T, N>*> new_precise_events;
        std::vector<Event<T, N>*> new_rough_events;
        for (size_t i=0; i<events.size(); i++){
            if (events[i]->is_precise()){
                new_precise_events.push_back(events[i]->clone());
            }
            else{
                new_rough_events.push_back(events[i]->clone());
            }
        }

        //push the pointers into a new (sorted) array
        std::vector<Event<T, N>*> result(events.size());
        std::copy(new_precise_events.begin(), new_precise_events.end(), result.begin());
        std::copy(new_rough_events.begin(), new_rough_events.end(), result.begin() + new_precise_events.size());

        for (Event<T, N>* ev : result){
            ev->set_args(this->_args);
        }

        //NOW we can delete our current events
        _delete_events();

        _events = result;        
    }

    void _delete_events(){
        for (size_t i=0; i<_events.size(); i++){
            delete this->_events[i];
            this->_events[i] = nullptr;
        }
    }

};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/

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


template<class T, int N>
bool OdeSolver<T, N>::advance(){
    State<T, N> next = adaptive_step();
    return _go_to_state(next);
}



template<class T, int N>
bool OdeSolver<T, N>::advance_by(const T& habs){
    if (habs <= 0){
        std::cout << std::endl << "Please provide a positive stepsize in .advance_by(habs)\n";
        return false;
    }

    if (_t_check != nullptr){
        _clear_checkpoint();
    }

    bool _set_non_stiff = false;
    if (habs <= _h_min && !_is_stiff){
        _set_non_stiff = true;
    }
    vec<T, N> q_next = step(_t, _q, habs*_direction);
    State<T, N> next = {_t+habs*_direction, q_next, habs};
    bool success = _go_to_state(next);
    if (success && _set_non_stiff){
        _is_stiff = false;
    }
    return success;
}

template<class T, int N>
bool OdeSolver<T, N>::advance_by_any(const T& h){
    if (_t_check != nullptr){
        _clear_checkpoint();
    }

    set_goal(_t+h);
    vec<T, N> q_next = step(_t, _q, h);
    State<T, N> next = {_t+h, q_next, h*_direction};
    return _go_to_state(next);
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
    if (_current_event_index == -1 && _t_check != nullptr){
        _clear_checkpoint();
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
            _q = this->step(_t, _q, _tmax-_t);
        }
        else{
            _q = this->step(_t, _q, _tmax-_t);
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
            write_chechpoint(_file, _t, q(), _current_event_index);
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
    std::function<vec<T, N>(T)> qfunc = [this](const T& t_next) -> vec<T, N> { return _q_step(t_next);};
    
    if (event.determine(this->_t, this->_q, next.t, next.q, qfunc)){
        if (_current_event_index == -1 && event.allows_checkpoint()){
            _make_checkpoint(next.t, next.q, next.h_next);
        }
        else if ( _t_check != nullptr && event.has_mask()){
            _clear_checkpoint();
        }
        next = {event.data().t(), event.data().q_true(), next.h_next};
        if (event.hide_mask()){
            _q_exposed = &event.data().q();
        }
        return true;
    }
    return false;
}

template<class T, int N>
bool OdeSolver<T, N>::_go_to_state(State<T, N>& next){

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
        if (_t_check != nullptr){
            next = {*_t_check, *_q_check, *_habs_check};
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



#endif



