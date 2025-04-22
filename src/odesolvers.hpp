#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "events.hpp"
#include <limits>



template<class Tt, class Ty>
struct SolverArgs{

    const Func<Tt, Ty> f;
    const Tt t0;
    const Ty q0;
    const Tt rtol;
    const Tt atol;
    const Tt h_min;
    const Tt h_max;
    const Tt first_step;
    const std::vector<Tt> args;
    const std::vector<AnyEvent<Tt, Ty>*> events;
    const Func<Tt, Ty> mask;
    const std::string save_dir;
    const bool save_events_only;
};


template<class Tt, class Ty>
void write_chechpoint(std::ofstream& file, const Tt& t, const Ty& q, const int& event_index){
    file << event_index << " " << std::setprecision(16) << t;
    for (size_t i=0; i<static_cast<size_t>(q.size()); i++){
        file << " " << std::setprecision(16) << q[i];
    }
    file << "\n";
}


template<class Tt, class Ty>
class OdeSolver{


public:

    using Callable = Func<Tt, Ty>;
    const Tt MAX_FACTOR = Tt(10);
    const Tt SAFETY = Tt(9)/10;
    const Tt MIN_FACTOR = Tt(2)/10;
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
    bool advance_by(const Tt& habs);
    bool advance_by_any(const Tt& h);
    bool advance();
    bool set_goal(const Tt& t_max);

    //ACCESSORS
    const Tt& t() const { return _t; }
    const Ty& q() const { return *_q_exposed; }
    const Ty& q_true() const { return _q; }

    const Tt& stepsize() const { return _habs; }
    const Tt& tmax() const { return _tmax; }
    const int& direction() const { return _direction; }
    const Tt& rtol() const { return _rtol; }
    const Tt& atol() const { return _atol; }
    const Tt& h_min() const { return _h_min; }
    const Tt& h_max() const { return _h_max; }
    const std::vector<Tt>& args() const { return _args; }
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
            return set_goal(-inf<Tt>());
        }
        else{
            return set_goal(inf<Tt>());
        }

    }

    const bool at_event()const{
        return _current_event_index != -1;
    }

    std::string event_name() const{
        return at_event() ? current_event()->name() : "";
    }

    const SolverState<Tt, Ty> state() const {
        return {_t, q(), _habs, event_name(), _diverges, _is_stiff, _is_running, _is_dead, _N, _message};
    }

    const AnyEvent<Tt, Ty>* current_event() const{
        //we need pointer and not reference, because it might be null
        return at_event() ? _events[_current_event_index] : nullptr;
    }

    const int& current_event_index() const{
        //we need pointer and not reference, because it might be null
        return _current_event_index;
    }

    const AnyEvent<Tt, Ty>* event(const size_t& i){
        return _events[i];
    }

    size_t events_size()const{
        return _events.size();
    }


    //MEMBER FUNCTIONS BELOW IMPLEMENTED BY CUSTOM DERIVED CLASSES
    //THEY FIRST 2 MUST NOT DEPEND ON THE CURRENT STATE

    virtual Ty step(const Tt& t_old, const Ty& q_old, const Tt& h) const = 0;

    virtual State<Tt, Ty> adaptive_step() const = 0; //derived implementation must account for h_min

    virtual OdeSolver<Tt, Ty>* clone() const = 0;

    Tt auto_step()const{
        //returns absolute value of emperically determined first step.
        if (_direction == 0){
            //needed even if the resulting stepsize will have a positive value.
            throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
        }
        Tt h0, d2, h1;
        Ty y1, f1;
        Ty scale = _atol + cwise_abs(_q)*_rtol;
        Ty _dq = _f(_t, _q, _args);
        Tt d0 = rms_norm((_q/scale).eval());
        Tt d1 = rms_norm((_dq/scale).eval());
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
            h1 = std::max(Tt(1e-6), 1e-3*h0);
        }
        else{
            h1 = pow(100*std::max(d1, d2), -Tt(1)/Tt(ERR_EST_ORDER+1));
        }

        return std::min({100*h0, h1, _h_max});
    }



protected:

    OdeSolver(const SolverArgs<Tt, Ty>& S, const int& order, const int& err_est_order): ORDER(order), ERR_EST_ORDER(err_est_order), _f(S.f), _t(S.t0), _q(S.q0), _rtol(S.rtol), _atol(S.atol), _h_min(S.h_min), _h_max(S.h_max), _args(S.args), _n(S.q0.size()), _filename(S.save_dir), _save_events_only(S.save_events_only){
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
            Func<Tt, Ty> f_tmp = S.f;
            Func<Tt, Ty> msk = S.mask;
            _f = [msk, f_tmp](const Tt& t, const Ty& q, const std::vector<Tt>& args)->Ty {
                return msk(t, f_tmp(t, q, args), args);
            };
        }
        _q_exposed = &_q;
        _make_new_events(S.events);
        set_goal(_t);
        if (!_filename.empty()){
            if (typeid(Tt) != typeid(_q[0])){
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

    OdeSolver(const OdeSolver<Tt, Ty>& other) : ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER){
        _copy_data(other);
        _filename = "";
        _autosave = false;
    };

    OdeSolver(OdeSolver<Tt, Ty>&& other): ORDER(other.ORDER), ERR_EST_ORDER(other.ERR_EST_ORDER), _file(std::move(other._file)){
        //not the most efficient, but the most readable :)
        //besides the time scale of a copy is insignificant to the timescale of
        //solving an ode.
        _copy_data(other);
    }


    OdeSolver<Tt, Ty>& operator=(const OdeSolver<Tt, Ty>& other){
        _copy_data(other);
        _filename = "";
        _autosave = false;
        return *this;
    }

    inline Ty f(const Tt& t, const Ty& q)const{
        return _f(t, q, _args);
    }



private:

    Callable _f;
    Tt _t;
    Ty _q;
    Tt _rtol;
    Tt _atol;
    Tt _h_min;
    Tt _h_max;
    Tt _habs;
    std::vector<Tt> _args;

    size_t _n; //size of ode system
    Tt _tmax;
    bool _diverges = false;
    bool _is_stiff = false;
    bool _is_running = true;
    bool _is_dead = false;
    size_t _N=0;//total number of solution updates
    std::string _message; //different from "running".
    int _direction;
    std::vector<AnyEvent<Tt, Ty>*> _events;
    int _current_event_index = -1;
    std::string _filename;
    std::ofstream _file;
    bool _autosave = false;
    bool _save_events_only = false;
    Tt* _t_check = nullptr;
    Ty* _q_check = nullptr;
    Tt* _habs_check = nullptr;
    const Ty* _q_exposed = nullptr; //view_only pointer

    Ty _q_step(const Tt& t_next)const{
        return this->step(this->_t, this->_q, t_next-this->_t);
    }

    bool _adapt_to_event(State<Tt, Ty>& next, AnyEvent<Tt, Ty>& event);

    bool _go_to_state(State<Tt, Ty>& next);

    bool _update(const Tt& t_new, const Ty& y_new, const Tt& h_next);

    void _warn_dead(){
        std::cout << std::endl << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << "\n";
    }

    void _warn_paused(){
        std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
    }

    void _warn_travolta(){
        std::cout << std::endl << "Solver has not been specified an integration direction, possibly because the Tmax goal was reached. Please set a new Tmax goal first or free() the solver.\n";
    }

    void _copy_data(const OdeSolver<Tt, Ty>& other){
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
            _q_exposed = &current_event()->q_event();
        }
    }

    void _make_checkpoint(const Tt& t, const Ty& q, const Tt& habs){
        _t_check = new Tt;
        _q_check = new Ty;
        _habs_check = new Tt;
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

    void _make_new_events(const std::vector<AnyEvent<Tt, Ty>*>& events){

        //FIRST create a new vector with new allocated objects, because "events" might be
        //our current _events vector. We sort the vector to contain normal events first,
        //and stop_events after to improve runtime performance and not miss out on any stop_events
        //if a single step encouters multiple events.
        std::vector<AnyEvent<Tt, Ty>*> new_events;
        std::vector<AnyEvent<Tt, Ty>*> new_stop_events;
        for (size_t i=0; i<events.size(); i++){
            if (events[i]->is_stop_event()){
                new_stop_events.push_back(events[i]->clone());
            }
            else{
                new_events.push_back(events[i]->clone());
            }
        }

        //push the pointers into a new (sorted) array
        std::vector<AnyEvent<Tt, Ty>*> result(events.size());
        std::copy(new_events.begin(), new_events.end(), result.begin());
        std::copy(new_stop_events.begin(), new_stop_events.end(), result.begin() + new_events.size());

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

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::set_goal(const Tt& t_max_new){
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


template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::advance(){
    State<Tt, Ty> next = adaptive_step();
    return _go_to_state(next);
}



template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::advance_by(const Tt& habs){
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
    Ty q_next = step(_t, _q, habs*_direction);
    State<Tt, Ty> next = {_t+habs*_direction, q_next, habs};
    bool success = _go_to_state(next);
    if (success && _set_non_stiff){
        _is_stiff = false;
    }
    return success;
}

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::advance_by_any(const Tt& h){
    if (_t_check != nullptr){
        _clear_checkpoint();
    }

    set_goal(_t+h);
    Ty q_next = step(_t, _q, h);
    State<Tt, Ty> next = {_t+h, q_next, h*_direction};
    return _go_to_state(next);
}

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_update(const Tt& t_new, const Ty& y_new, const Tt& h_next){
    
    bool success = true;
    if (h_next < 0){//h_next is always positive, it is the absolute value of the true stepsize
        success = false;
        throw std::runtime_error("Bug detected: Absolute stepsize < 0");
    }

    if (!All_isFinite(y_new)){
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

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_adapt_to_event(State<Tt, Ty>& next, AnyEvent<Tt, Ty>& event){
    // takes next state (which means tnew, hnew, and hnext_new)
    // if it is not an event or smth it is left unchanged.
    // otherwise, it is modified to depict the event with high accuracy
    std::function<Ty(Tt)> qfunc = [this](const Tt& t_next) -> Ty { return _q_step(t_next);};
    
    if (event.determine(this->_t, this->_q, next.t, next.q, this->_args, qfunc)){
        if (_current_event_index == -1 && event.allows_checkpoint()){
            _make_checkpoint(next.t, next.q, next.h_next);
        }
        else if ( _t_check != nullptr && event.has_mask()){
            _clear_checkpoint();
        }
        next = {event.t_event(), event.q_true_event(), next.h_next};
        if (event.hide_mask()){
            _q_exposed = &event.q_event();
        }
        return true;
    }
    return false;
}

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_go_to_state(State<Tt, Ty>& next){

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
                if (current_event()->is_stop_event()){
                    break;
                }
            }
        }

        return _update(next.t, next.q, next.h_next);
    }

}



#endif



