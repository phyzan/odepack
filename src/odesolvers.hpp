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

    //arguments below almost identical to solver args
    const Callable f;
    const Tt rtol;
    const Tt atol;
    const Tt h_min;
    const std::vector<Tt> args;
    const Tt event_tol;
    const size_t n; //size of ode system

    const Tt MAX_FACTOR = Tt(10);
    const Tt SAFETY = Tt(9)/10;
    const Tt MIN_FACTOR = Tt(2)/10;

private:

    //arguments below are passed into the SolverState when commanded
    Tt _t;
    Ty _q;
    Tt _habs;
    Tt _tmax;
    bool _diverges = false;
    bool _is_stiff = false;
    bool _is_running = true;
    bool _is_dead = false;
    size_t _N=0;//total number of solution updates
    std::string _message = "Alive"; //different from "running".
    int _direction;
    const std::vector<StopEvent<Tt, Ty>> _stop_events;
    std::vector<Event<Tt, Ty>> _events; //we make a copy of the provided vector
    const Event<Tt, Ty>* _current_event = nullptr;
    int _current_event_index = -1;


public:

    virtual ~OdeSolver() = default;

    //MODIFIERS
    void stop(const std::string& text = "") {_is_running = false; _message = (text == "") ? "Stopped by user" : text;}
    void kill(const std::string& text = "") {_is_running = false; _is_dead = true; _message = (text == "") ? "Killed by user" : text;}
    bool advance_by(const Tt& h);
    bool advance();
    void set_goal(const Tt& t_max);

    //ACCESSORS
    const Tt& t = _t;
    const Ty& q = _q;
    const Tt& stepsize = _habs;
    const Tt& tmax = _tmax;
    const int& direction = _direction;
    const bool at_event()const{
        return _current_event != nullptr;
    }
    std::string event_name() const{
        return at_event() ? _current_event->name() : "";
    }
    const bool& diverges() const {return _diverges;}
    const bool& is_stiff() const {return _is_stiff;}
    const bool& is_running() const {return _is_running;}
    const bool& is_dead() const {return _is_dead;}
    const std::string& message() {return _message;}
    const SolverState<Tt, Ty> state() const {
        return {_t, _q, _habs, event_name(), _diverges, _is_stiff, _is_running, _is_dead, _N, _message};
    }

    const Event<Tt, Ty>* current_event() const{
        //we need pointer and not reference, because it might be null
        return _current_event;
    }

    const int& current_event_index() const{
        //we need pointer and not reference, because it might be null
        return _current_event_index;
    }

    const std::vector<Event<Tt, Ty>>& events() const{
        return _events;
    }


    //MEMBER FUNCTIONS BELOW IMPLEMENTED BY CUSTOM DERIVED CLASSES
    //THEY MUST NOT DEPEND ON THE CURRENT STATE

    virtual Ty step(const Tt& t_old, const Ty& q_old, const Tt& h) const = 0;

    virtual State<Tt, Ty> adaptive_step() const = 0;



protected:

    OdeSolver(const SolverArgs<Tt, Ty>& S): f(S.f), rtol(S.rtol), atol(S.atol), h_min(S.h_min), args(S.args), event_tol(S.event_tol), n(S.q0.size()), _t(S.t0), _q(S.q0), _habs(S.habs), _stop_events(S.stop_events), _events(S.events) {
        set_goal(S.tmax);
    }


private:

    OdeSolver operator=(const OdeSolver&) = delete;
    OdeSolver(const OdeSolver& other) = default;

    bool _adapt_to_event(State<Tt, Ty>& next, Event<Tt, Ty>& event);

    bool _go_to_state(State<Tt, Ty>& next);

    bool _update(const Tt& t_new, const Ty& y_new, const Tt& h_next);

    void _warn_dead(){
        throw std::runtime_error("Solver has permanently stop integrating. If this is not due to calling .kill(), call state() to see the cause.");
    }
};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/

template<class Tt, class Ty>
void OdeSolver<Tt, Ty>::set_goal(const Tt& t_max_new){
    if ((_is_stiff || _diverges) && (!_is_dead || _is_running) ){
        //sanity check. 
        throw std::runtime_error("Bug detected");
    }

    if (_is_dead){
        _warn_dead();
    }
    else if (t_max_new == _t){
        _direction = 0;
        _tmax = t_max_new;
        stop("Waiting for new Tmax");
    }
    else{
        _tmax = t_max_new;
        _is_running = true;
        _direction = ( t_max_new > _t) ? 1 : -1;
    }
}


template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::advance(){
    State<Tt, Ty> next = adaptive_step();
    return _go_to_state(next);
}



template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::advance_by(const Tt& habs){
    Ty q_next = step(habs*direction);
    return _go_to_state({_t+habs*direction, q_next, habs});
}


template<class Tt, class Ty>
bool OdeSolver<Tt, Ty>::_update(const Tt& t_new, const Ty& y_new, const Tt& h_next){
    
    bool success = true;
    if (_is_dead){
        success = false;
        _warn_dead();
    }
    else if (! _is_running){
        success = false;
        throw std::runtime_error("Solver has finished integrating. Please set new t_max goal to continue integrating *before* advancing");
    }

    if (h_next < 0){//h_next is always positive, it is the absolute value of the true stepsize
        success = false;
        throw std::runtime_error("Bug detected");
    }

    if (!All_isFinite(y_new)){
        kill("Ode solution diverges");
        _diverges = true;
        success = false;
    }
    else if (h_next <= h_min){
        kill("Ode very stiff");
        _is_stiff = true;
        success = false;
    }
    else if (t_new*direction >= _tmax*direction){
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
        _message = "Alive";
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
    
    if (event.determine(this->_t, next.t, this->args, qfunc, this->event_tol)){
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

    if (_N > 0){
        _current_event = nullptr;
        _current_event_index = -1;
        bool success;
        int k = 0;

        for (const StopEvent<Tt, Ty>& _stop_ev : _stop_events){
            if (_stop_ev.is_between(this->_t, this->_q, next.t, next.q, this->args)){
                success = _update(next.t, next.q, next.h_next);
                stop(_stop_ev.name());
                return success;
            }
        }

        for (Event<Tt, Ty>& ev : _events){
            if (_adapt_to_event(next, ev)){
                success = _update(ev.t_event(), ev.q_event(), next.h_next);
                _current_event = &ev;
                _current_event_index = k;
                return success;
            }
            k++;
        }
    }
    return _update(next.t, next.q, next.h_next);
}



#endif



