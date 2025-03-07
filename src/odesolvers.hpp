#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "tools.hpp"
#include <limits>


template<class Tt, class Ty>
class OdeSolver{


public:

    using Callable = Func<Tt, Ty>;
    const Tt MAX_FACTOR = Tt(10);
    const Tt SAFETY = Tt(9)/10;
    const Tt MIN_FACTOR = Tt(2)/10;


    virtual ~OdeSolver() = default;

    //MODIFIERS
    void stop(const std::string& text = "") {_is_running = false; _message = (text == "") ? "Stopped by user" : text;}
    void kill(const std::string& text = "") {_is_running = false; _is_dead = true; _message = (text == "") ? "Killed by user" : text;}
    bool advance_by(const Tt& h);
    bool advance();
    bool set_goal(const Tt& t_max);

    //ACCESSORS
    const Tt& t() const { return _t; }
    const Ty& q() const { return _q; }
    const Tt& stepsize() const { return _habs; }
    const Tt& tmax() const { return _tmax; }
    const int& direction() const { return _direction; }
    const Callable& f() const { return _f; }
    const Tt& rtol() const { return _rtol; }
    const Tt& atol() const { return _atol; }
    const Tt& h_min() const { return _h_min; }
    const Tt& event_tol() const { return _event_tol; }
    const std::vector<Tt>& args() const { return _args; }
    const size_t& Nsys() const { return _n; }
    const bool& diverges() const {return _diverges;}
    const bool& is_stiff() const {return _is_stiff;}
    const bool& is_running() const {return _is_running;}
    const bool& is_dead() const {return _is_dead;}
    const std::string& message() {return _message;}

    bool resume(){
        if (_is_dead){
            _warn_dead();
        }
        else if (!_is_running) {
            _warn_paused();
        }
        else{
            _message = "Running";
            _is_running = true;
            return true;
        }
        return false;
    }

    bool free(){
        return _solver->set_goal(std::numeric_limits<Tt>::infinity());
    }

    const bool at_event()const{
        return _current_event_index != -1;
    }

    std::string event_name() const{
        return at_event() ? _events[_current_event_index].name() : "";
    }

    const SolverState<Tt, Ty> state() const {
        return {_t, _q, _habs, event_name(), _diverges, _is_stiff, _is_running, _is_dead, _N, _message};
    }

    const Event<Tt, Ty>* current_event() const{
        //we need pointer and not reference, because it might be null
        return (_current_event_index == -1) ? nullptr : &(_events[_current_event_index]);
    }

    const int& current_event_index() const{
        //we need pointer and not reference, because it might be null
        return _current_event_index;
    }

    const std::vector<Event<Tt, Ty>>& events() const{
        return _events;
    }


    //MEMBER FUNCTIONS BELOW IMPLEMENTED BY CUSTOM DERIVED CLASSES
    //THEY FIRST 2 MUST NOT DEPEND ON THE CURRENT STATE

    virtual Ty step(const Tt& t_old, const Ty& q_old, const Tt& h) const = 0;

    virtual State<Tt, Ty> adaptive_step() const = 0;

    virtual OdeSolver<Tt, Ty>* clone() const = 0;



protected:

    OdeSolver(const SolverArgs<Tt, Ty>& S): _f(S.f), _rtol(S.rtol), _atol(S.atol), _h_min(S.h_min), _args(S.args), _event_tol(S.event_tol), _n(S.q0.size()), _t(S.t0), _q(S.q0), _habs(S.habs), _stop_events(S.stop_events), _events(S.events) {
        set_goal(S.tmax);
    }

    OdeSolver<Tt, Ty>& operator=(const OdeSolver<Tt, Ty>& other){
        _copy_data(other);
    }

    OdeSolver(const OdeSolver<Tt, Ty>& other){
        _copy_data(other);
    };


private:
    Callable _f;
    Tt _rtol;
    Tt _atol;
    Tt _h_min;
    std::vector<Tt> _args;
    Tt _event_tol;
    size_t _n; //size of ode system
    Tt _t;
    Ty _q;
    Tt _habs;
    Tt _tmax;
    bool _diverges = false;
    bool _is_stiff = false;
    bool _is_running = true;
    bool _is_dead = false;
    size_t _N=0;//total number of solution updates
    std::string _message; //different from "running".
    int _direction;
    std::vector<StopEvent<Tt, Ty>> _stop_events;
    std::vector<Event<Tt, Ty>> _events;
    int _current_event_index = -1;



    bool _adapt_to_event(State<Tt, Ty>& next, Event<Tt, Ty>& event);

    bool _go_to_state(State<Tt, Ty>& next);

    bool _update(const Tt& t_new, const Ty& y_new, const Tt& h_next);

    void _warn_dead(){
        std::cout << std::endl << "Solver has permanently stop integrating. If this is not due to calling .kill(), call state() to see the cause.\n";
    }

    void _warn_paused(){
        std::cout << std::endl << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so.\n";
    }

    void _copy_data(const OdeSolver<Tt, Ty>& other){
        //arguments below are passed into the SolverState when commanded
        _q = other._q;
        _habs = other._habs;
        _tmax = other._tmax;
        _diverges = other._diverges;
        _is_stiff = other._is_stiff;
        _is_running = other._is_running;
        _is_dead = other._is_dead;
        _N = other._N;
        _message = other._message;
        _direction = other._direction;
        _stop_events = other._stop_events; // Copying the vector
        _events = other._events; // Copying the vector
        _current_event_index = other._current_event_index;
        _f = other._f;
        _rtol = other._rtol;
        _atol = other._atol;
        _h_min = other._h_min;
        _args = other._args;
        _event_tol = other._event_tol;
        _n = other._n;
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
    if ((_is_stiff || _diverges) && (!_is_dead || _is_running) ){
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
    Ty q_next = step(habs*_direction);
    return _go_to_state({_t+habs*_direction, q_next, habs});
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
    else if (h_next <= _h_min){
        kill("Ode very stiff");
        _is_stiff = true;
        success = false;
    }
    else if (t_new*_direction >= _tmax*_direction){
        stop("T_max goal reached");
        _q = this->step(_t, _q, _tmax-_t);
        _t = _tmax;
        _habs = h_next;
        _N++;
    }
    else{
        _t = t_new;
        _q = y_new;
        _habs = h_next;
        _N++;
    }

    return success;
}

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_adapt_to_event(State<Tt, Ty>& next, Event<Tt, Ty>& event){
    // takes next state (which means tnew, hnew, and hnext_new)
    // if it is not an event or smth it is left unchanged.
    // otherwise, it is modified to depict the event with high accuracy
    std::function<Ty(Tt)> qfunc;
    Tt t_new, h_new;
    Ty q_new;

    qfunc = [this](const Tt& t_next) -> Ty { return this->step(this->_t, this->_q, t_next-this->_t);};
    
    if (event.determine(this->_t, next.t, this->_args, qfunc, this->_event_tol)){
        t_new = event.t_event();
        q_new = event.q_event();
        h_new = t_new - this->_t;
        next = {t_new, q_new, abs(h_new)};
        return true;
    }
    return false;
}

template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_go_to_state(State<Tt, Ty>& next){

    if (_is_dead){
        _warn_dead();
    }
    else if (!_is_running){
        _warn_paused();
    }
    else if (_N > 0){
        _current_event_index = -1;
        bool success;
        int k = 0;

        for (const StopEvent<Tt, Ty>& _stop_ev : _stop_events){
            if (_stop_ev.is_between(this->_t, this->_q, next.t, next.q, this->_args)){
                success = _update(next.t, next.q, next.h_next);
                stop(_stop_ev.name());
                return success;
            }
        }

        for (Event<Tt, Ty>& ev : _events){
            if (_adapt_to_event(next, ev)){
                success = _update(ev.t_event(), ev.q_event(), next.h_next);
                _current_event_index = k;
                return success;
            }
            k++;
        }
    }
    return _update(next.t, next.q, next.h_next);
}



#endif



