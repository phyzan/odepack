#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include "tools.hpp"

template<class Tt, class Ty, bool raw_ode, bool raw_event>
struct SolverArgs{

    const ode_t<Tt, Ty, raw_ode> f;
    const ICS<Tt, Ty> ics;
    const Tt t;
    const Tt h;
    const Tt rtol;
    const Tt atol;
    const Tt h_min;
    const std::vector<Tt> args;
    const event_t<Tt, Ty, raw_event> getevent;
    const event_t<Tt, Ty, raw_event> stopevent;
    
};

template<class Tt, class Ty, bool raw_ode, bool raw_event>
SolverArgs<Tt, Ty, raw_ode, raw_event> to_SolverArgs(const ode_t<Tt, Ty, raw_ode>& f, const OdeArgs<Tt, Ty, raw_event>& args){
    return {f, args.ics, args.t, args.h, args.rtol, args.atol, args.cutoff_step, args.args, args.getcond, args.breakcond};
}


template<class Tt, class Ty, bool raw_ode, bool raw_event>
class OdeSolver{


public:

    using Callable = ode_t<Tt, Ty, raw_ode>;

    const Tt MAX_FACTOR = Tt(10);
    const Tt SAFETY = Tt(9)/10;
    const Tt MIN_FACTOR = Tt(2)/10;

    const Callable f;
    const Tt t_max;
    const Tt min_h;
    const std::vector<Tt> args;
    const int direction;
    const size_t n;
    const Tt rtol;
    const Tt atol;
    event_t<Tt, Ty, raw_event> getevent;
    event_t<Tt, Ty, raw_event> stopevent;

private:

    Tt _h;
    Tt _t;
    Ty _y;
    bool _is_running = true;
    bool _diverges = false;
    bool _is_stiff = false;
    size_t neval=0;
    bool _event = false;

public:

    virtual ~OdeSolver() = default;

    //ACCESSORS
    const Tt& t_now() const {return _t;}

    const Ty& y_now() const {return _y;}

    const Tt& h_now() const {return _h;}

    const bool& event() const {return _event;}

    const bool& is_running() const {return _is_running;}

    SolverState<Tt, Ty> state() const {
        SolverState<Tt, Ty> res = {_t, _y, _diverges, _is_stiff, _is_running, neval, _event};
        return res;
    }

    //MODIFIERS
    void stop() {_is_running = false;}

    bool advance_by(const Tt& h);

    bool advance();

    //MEMBER FUNCTIONS BELOW IMPLEMENTED BY CUSTOM DERIVED CLASSES
    //THEY MUST NOT DEPEND ON THE CURRENT STATE

    virtual Ty step(const Tt& t_old, const Ty& y_old, const Tt& h) const = 0;

    virtual State<Tt, Ty> adaptive_step() const = 0;



protected:

    OdeSolver(const SolverArgs<Tt, Ty, raw_ode, raw_event>& S): f(S.f), t_max(S.t), min_h(S.h_min), args(S.args), direction( S.h > 0 ? 1 : -1), n(S.ics.y0.size()), rtol(S.rtol), atol(S.atol), getevent(S.getevent), stopevent(S.stopevent), _h(S.h), _t(S.ics.t0), _y(S.ics.y0) {}

    bool _update(const Tt& t_new, const Ty& y_new, const Tt& h_next, const bool& is_event);

private:

    OdeSolver operator=(const OdeSolver&) = delete;
    OdeSolver(const OdeSolver& other) = default;

    bool _examine_state(State<Tt, Ty>& next, const event_t<Tt, Ty, raw_event>& event)const;

    bool _go_to_state(State<Tt, Ty>& next);
};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/


template<class Tt, class Ty, bool raw_ode, bool raw_event>
bool OdeSolver<Tt, Ty, raw_ode, raw_event>::advance(){
    State<Tt, Ty> next = adaptive_step();
    return _go_to_state(next);
}



template<class Tt, class Ty, bool raw_ode, bool raw_event>
bool OdeSolver<Tt, Ty, raw_ode, raw_event>::advance_by(const Tt& h){
    State<Tt, Ty> next = step(h);
    return _go_to_state(next);
}


template<class Tt, class Ty, bool raw_ode, bool raw_event>
bool OdeSolver<Tt, Ty, raw_ode, raw_event>::_update(const Tt& t_new, const Ty& y_new, const Tt& h_next, const bool& is_event){

    bool success = true;
    if (! _is_running){
        success = false;
        throw std::runtime_error("Solver has finished integrating.");
    }

    bool _stop = false;
    Tt stepsize = h_next*direction;
    if (stepsize < 0){
        success = false;
        throw std::runtime_error("Wrong direction of integration");
    }

    if (stepsize <= min_h){
        _is_stiff = true;
        _stop = true;
        success = false;
    }
    if (!All_isFinite(y_new)){
        _diverges = true;
        _stop = true;
        success = false;
    }
    else if (t_new*direction >= t_max*direction){
        _y = this->step(_t, _y, t_max-_t);
        _t = t_max;
        _h = h_next;
        neval++;
        _event = is_event;
        _stop = true;
    }
    else{
        _t = t_new;
        _y = y_new;
        _h = h_next;
        neval++;
        _event = is_event;
    }

    if (_stop) stop();

    return success;
}

template<class Tt, class Ty, bool raw_ode, bool raw_event>
bool OdeSolver<Tt, Ty, raw_ode, raw_event>::_examine_state(State<Tt, Ty>& next, const event_t<Tt, Ty, raw_event>& event)const{
        
    Tt t_new, h_new;
    Ty y_new;

    if ( event(_t, _y, next.t, next.y) ){
        std::function<Tt(Tt)> func = [this, event](const Tt& t_next) -> int {
            Ty y_next = this->step(this->_t, this->_y, t_next-this->_t);
            return (event(this->_t, this->_y, t_next, y_next) > 0) ? 1: -1;
        };
        
        t_new = bisect(func, _t, next.t, 1e-12)[2];
        h_new = t_new - _t;
        y_new = step(_t, _y, h_new);
        next = {t_new, y_new, h_new};
        return true;
    }
    else{
        return false;
    }
}

template<class Tt, class Ty, bool raw_ode, bool raw_event>
bool OdeSolver<Tt, Ty, raw_ode, raw_event>::_go_to_state(State<Tt, Ty>& next){
    bool is_event = false;
    if (stopevent != nullptr && _examine_state(next, stopevent)){
        bool res = _update(next.t, next.y, next.dt, false);
        stop();
        return res;
    }
    if (getevent != nullptr){
        is_event = _examine_state(next, getevent);
    }
    return _update(next.t, next.y, next.dt, is_event);
}



#endif
