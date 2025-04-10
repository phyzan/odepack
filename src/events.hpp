#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include "tools.hpp"

template<class Tt, class Ty>
using event_f = std::function<Tt(const Tt&, const Ty&, const std::vector<Tt>&)>;

template<class Tt, class Ty>
using is_event_f = std::function<bool(const Tt&, const Ty&, const std::vector<Tt>&)>;


template<class Tt, class Ty>
class AnyEvent{

/**
 * @brief Base class representing an event that might be encoutered during the integration of an ODE.
 */

public:

    virtual bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::vector<Tt>& args, const std::function<Ty(const Tt&)>& q) = 0;
    /**
     * @brief If an event is encoutered between two integration steps, this member function determines exactly when that occurs. The event time and function value
     * 
     * 
     */

    virtual void go_back(){ _clear();}

    virtual AnyEvent<Tt, Ty>* clone() const = 0;

    const Tt& t_event()const{ return *_t_event;}

    const Ty& q_event()const{ return *_q_event;}

    const Ty& q_true_event()const{ return *_q_masked;}

    const std::string& name()const{ return _name;}

    const bool& hide_mask()const{ return _hide_mask;}

    bool has_mask()const{ return _mask != nullptr;}

    virtual bool is_stop_event()const {return false;}

    bool allows_checkpoint() const{
        //!is_stop_event() is not a strict condition, but saves time
        //by not making a checkpoint since the odesolver will step on the
        //next step anyway.
        return !has_mask() && !is_stop_event();
    }

    virtual ~AnyEvent(){ _clear();}

protected:

    std::string _name;
    event_f<Tt, Ty> _when = nullptr;
    is_event_f<Tt, Ty> _check_if = nullptr;
    Func<Tt, Ty> _mask = nullptr;
    bool _hide_mask; // variable that is only used externally for odesolvers to determine when and whether to call q_event or q_masked.
    Tt* _t_event = nullptr;
    Ty* _q_event = nullptr;
    Ty* _q_masked = nullptr;


    AnyEvent(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if, Func<Tt, Ty> mask, const bool& hide_mask): _name(name), _when(when), _check_if(check_if), _mask(mask), _hide_mask(hide_mask){
        if (name == ""){
            throw std::runtime_error("Please provide a non empty name when instanciating an Event-related class");
        }
    }

    AnyEvent(const AnyEvent<Tt, Ty>& other): _name(other._name), _when(other._when), _check_if(other._check_if), _mask(other._mask), _hide_mask(other._hide_mask){
        _copy_event_data(other);
    }

    AnyEvent(AnyEvent<Tt, Ty>&& other): _name(std::move(other._name)), _when(std::move(other._when)), _check_if(std::move(other._check_if)), _mask(std::move(other._mask)), _hide_mask(other._hide_mask), _t_event(other._t_event), _q_event(other._q_event), _q_masked(other._q_masked){
        other._t_event = nullptr;
        other._q_event = nullptr;
        other._q_masked = nullptr;
    }

    AnyEvent<Tt, Ty>& operator=(const AnyEvent<Tt, Ty>& other){
        _name = other._name;
        _when = other._when;
        _check_if = other._check_if;
        _mask = other._mask;
        _hide_mask = other._hide_mask;
        _copy_event_data(other);
        return *this;
    }

    void _clear(){
        delete _t_event;
        delete _q_event;
        if (_q_masked != _q_event){
            delete _q_masked;
        }
        _t_event = nullptr;
        _q_event = nullptr;
        _q_masked = nullptr;
    }

    void _realloc(){
        //always used after _clear();
        _t_event = new Tt;
        _q_event = new Ty;
        if (_mask != nullptr){
            _q_masked = new Ty;
        }
        else{
            _q_masked = _q_event;
        }
    }

    void _set(const Tt& t, const Ty& q, const std::vector<Tt>& args){
        //always called right after _realloc(), before calling _clear() e.g.;
        *_t_event = t;
        *_q_event = q;
        if (_mask != nullptr){
            //this also means that _q_masked already points to a different memory location from _realloc();
            *_q_masked = _mask(*_t_event, q, args);
        }
    }

    void _copy_event_data(const AnyEvent<Tt, Ty>& other){
        _clear();
        if (other._t_event != nullptr){
            _realloc();
            *_t_event = *other._t_event;
            *_q_event = *other._q_event;
            if (_mask != nullptr){
                *_q_masked = *other._q_masked;
            }
        }
    }

};

template<class Tt, class Ty>
class Event : public AnyEvent<Tt, Ty>{

public:

    Event(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false, const Tt& event_tol=1e-12): AnyEvent<Tt, Ty>(name, when, check_if, mask, hide_mask), _event_tol(event_tol){
        _assert_func(when);
    }

    Event(const Event<Tt, Ty>& other):AnyEvent<Tt, Ty>(other), _event_tol(other._event_tol){}

    Event<Tt, Ty>& operator=(const Event<Tt, Ty>& other){
        if (&other != this){
            _event_tol = other._event_tol;
            AnyEvent<Tt, Ty>::operator=(other);
        }
        return *this;
    }

    Event<Tt, Ty>* clone() const override{
        return new Event<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::vector<Tt>& args, const std::function<Ty(const Tt&)>& q) override {
        this->_clear();
        Tt t_determined = t2;
        bool determined = false;
        if (this->_check_if == nullptr || (this->_check_if(t1, q1, args) && this->_check_if(t2, q2, args))){
            _ObjFun<Tt> obj_fun = [this, q, args](const Tt& t) ->Tt {
                return this->_when(t, q(t), args);
            };
            Tt val1 = this->_when(t1, q1, args);
            Tt val2 = this->_when(t2, q2, args);
            if (val1 * val2 <= 0 && val1 != 0){
                t_determined = bisect(obj_fun, t1, t2, this->_event_tol)[2];
                determined = true;
            }
        }

        if (determined){
            this->_realloc();
            this->_set(t_determined, q(t_determined), args);
        }
        return determined;

    }

private:

    Tt _event_tol;

};


template<class Tt, class Ty>
class PeriodicEvent : public AnyEvent<Tt, Ty>{

public:

    PeriodicEvent(const PeriodicEvent<Tt, Ty>& other):AnyEvent<Tt, Ty>(other), _period(other._period), _start(other._start), _np(other._np), _np_previous(other._np_previous){}

    PeriodicEvent(const std::string& name, const Tt& period, const Tt& start=0, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false): AnyEvent<Tt ,Ty>(name, nullptr, nullptr, mask, hide_mask), _period(period), _start(start){
        if (period <= 0){
            throw std::runtime_error("Period in periodic event must be positive. If integrating backwards, events are still counted.");
        }
    }

    PeriodicEvent<Tt, Ty>& operator=(const PeriodicEvent<Tt, Ty>& other){
        if (&other != this){
            _period = other._period;
            _start = other._start;
            _np = other._np;
            _np_previous = other._np_previous;
            _has_started = other._has_started;
            AnyEvent<Tt, Ty>::operator=(other);
        }
        return *this;

    }

    PeriodicEvent<Tt, Ty>* clone() const override{
        return new PeriodicEvent<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::vector<Tt>& args, const std::function<Ty(const Tt&)>& q) override {
        this->_clear();
        const int direction = (t2 > t1) ? 1 : -1;
        const Tt next = _has_started ? _start+(_np+direction)*_period : _start;
        if ( (t2*direction >= next*direction) && (next*direction > t1*direction) ){
            _np_previous = _np;
            _np += direction*this->_has_started;
            this->_realloc();
            this->_set(next, q(next), args);
            this->_has_started = true;
            return true;
        }
        else{
            return false;
        }
    }

    void go_back() override {
        this->_clear();
        if (_np != _np_previous){
            _np = _np_previous;
            if (_np_previous == 0){
                _has_started = false;
            }
        }
    }

private:
    Tt _period;
    Tt _start;
    long int _np = 0;
    long int _np_previous = 0;
    bool _has_started = false;

};


template<class Tt, class Ty>
class StopEvent : public AnyEvent<Tt, Ty>{

public:

    StopEvent(const StopEvent<Tt, Ty>& other) : AnyEvent<Tt, Ty>(other){}

    StopEvent(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false): AnyEvent<Tt ,Ty>(name, when, check_if, mask, hide_mask){
        _assert_func(when);
    }

    StopEvent<Tt, Ty>& operator=(const StopEvent<Tt, Ty>& other){
        if (&other != this){
            AnyEvent<Tt, Ty>::operator=(other);
        }
        return *this;

    }

    StopEvent<Tt, Ty>* clone() const override{
        return new StopEvent<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::vector<Tt>& args, const std::function<Ty(const Tt&)>& q) override {
        this->_clear();
        if (this->_check_if == nullptr || (this->_check_if(t1, q1, args) && this->_check_if(t2, q2, args))){
            Tt val1 = this->_when(t1, q1, args);
            Tt val2 = this->_when(t2, q2, args);
            if (val1 * val2 <= 0 && val1 != 0){
                this->_realloc();
                this->_set(t2, q2, args);
                return true;
            }
        }
        return false;
    }

    bool is_stop_event() const override{
        return true;
    }

};


#endif