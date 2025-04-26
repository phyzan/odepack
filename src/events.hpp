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
struct EventData{

    const Tt& t() const {return _t;}
    const Ty& q() const {return _q;}
    const Ty& q_true() const {return _q_masked;}

    EventData(){};

    EventData(const Tt& t, const Ty& q):_t(t), _q(q), _q_masked(q), _masked(false){}

    EventData(const Tt& t, const Ty& q, const Ty& q_true):_t(t), _q(q), _q_masked(q_true), _masked(true){}

private:
    Tt _t;
    Ty _q;
    Ty _q_masked;
    bool _masked;
};


template<class Tt, class Ty>
class Event{

public:

    virtual bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) = 0;

    virtual bool is_stop_event() const {return false;}

    virtual bool is_leathal() const { return false;}

    virtual bool check(const Tt& t, const Ty& q) const {return (_check_if == nullptr) ? true : _check_if(t, q, _args);}

    virtual void go_back();

    inline void set_args(const std::vector<Tt>& args){ _args = args;}

    virtual Event<Tt, Ty>* clone() const = 0;

    const EventData<Tt, Ty>& data() const;

    inline const std::string& name()const{ return _name;}

    inline const bool& hide_mask()const{ return _hide_mask;}

    inline bool has_mask()const{ return _mask != nullptr;}

    virtual bool allows_checkpoint() const{ return !has_mask(); }

    inline const size_t& counter() const{ return _counter; }

    virtual bool is_precise() const = 0;

    virtual ~Event(){delete _data;}


private:

    std::string _name;
    event_f<Tt, Ty> _when = nullptr;
    is_event_f<Tt, Ty> _check_if = nullptr;
    Func<Tt, Ty> _mask = nullptr;
    bool _hide_mask; // variable that is only used externally for odesolvers to determine when and whether to call q_event or q_masked.
    size_t _counter = 0;
    std::vector<Tt> _args = {};
    EventData<Tt, Ty>* _data = nullptr;
    

protected:

    Event(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if, Func<Tt, Ty> mask, const bool& hide_mask): _name(name), _when(when), _check_if(check_if), _mask(mask), _hide_mask(hide_mask){
        if (name == ""){
            throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
        }
    }

    Event(const Event<Tt, Ty>& other): _name(other._name), _when(other._when), _check_if(other._check_if), _mask(other._mask), _hide_mask(other._hide_mask), _counter(other._counter), _args(other._args){
        _copy_event_data(other);
    }

    Event(Event<Tt, Ty>&& other): _name(std::move(other._name)), _when(std::move(other._when)), _check_if(std::move(other._check_if)), _mask(std::move(other._mask)), _hide_mask(other._hide_mask), _counter(other._counter), _args(std::move(other._args)), _data(other._data){other._data = nullptr;}

    Event<Tt, Ty>& operator=(const Event<Tt, Ty>& other);

    inline Tt obj_fun(const Tt& t, const Ty& q) const { return _when(t, q, _args); }

    void _set(const Tt& t, const Ty& q);

    void _copy_event_data(const Event<Tt, Ty>& other);

    void _clear();

};

template<class Tt, class Ty>
class PreciseEvent : public Event<Tt, Ty>{

public:

    PreciseEvent(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false, const Tt& event_tol=1e-12): Event<Tt, Ty>(name, when, check_if, mask, hide_mask), _event_tol(event_tol){}

    PreciseEvent(const PreciseEvent<Tt, Ty>& other):Event<Tt, Ty>(other), _event_tol(other._event_tol){}

    PreciseEvent<Tt, Ty>& operator=(const PreciseEvent<Tt, Ty>& other);

    PreciseEvent<Tt, Ty>* clone() const override{
        return new PreciseEvent<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) override;

    inline bool is_precise() const final { return true;}

private:

    Tt _event_tol;

};


template<class Tt, class Ty>
class PeriodicEvent : public PreciseEvent<Tt, Ty>{

public:

    PeriodicEvent(const PeriodicEvent<Tt, Ty>& other):PreciseEvent<Tt, Ty>(other), _period(other._period), _start(other._start), _np(other._np), _np_previous(other._np_previous){}

    PeriodicEvent(const std::string& name, const Tt& period, const Tt& start=0, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false): PreciseEvent<Tt ,Ty>(name, nullptr, nullptr, mask, hide_mask, 0), _period(period), _start(start){
        if (period <= 0){
            throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
        }
    }

    PeriodicEvent<Tt, Ty>& operator=(const PeriodicEvent<Tt, Ty>& other);

    PeriodicEvent<Tt, Ty>* clone() const override{
        return new PeriodicEvent<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) override;

    void go_back() override;

private:
    Tt _period;
    Tt _start;
    long int _np = 0;
    long int _np_previous = 0;
    bool _has_started = false;

};


template<class Tt, class Ty>
class RoughEvent : public Event<Tt, Ty>{

public:

    RoughEvent(const RoughEvent<Tt, Ty>& other) : Event<Tt, Ty>(other){}

    RoughEvent(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false): Event<Tt ,Ty>(name, when, check_if, mask, hide_mask){}

    RoughEvent<Tt, Ty>& operator=(const RoughEvent<Tt, Ty>& other);

    RoughEvent<Tt, Ty>* clone() const override{
        return new RoughEvent<Tt, Ty>(*this);
    }

    bool determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) final ;

    inline bool allows_checkpoint() const override {return false; }

    inline bool is_precise() const final { return false;}

};











//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//----------------------------------IMPLEMENTATIONS--------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------










//Event CLASS

template<class Tt, class Ty>
void Event<Tt, Ty>::go_back(){
    if (_data != nullptr){
        _clear();
        _counter--;
    }
}

template<class Tt, class Ty>
const EventData<Tt, Ty>& Event<Tt, Ty>::data() const {
    if (_data == nullptr){
        throw std::runtime_error("Event has not been determined");
    }
    return *_data;
}

template<class Tt, class Ty>
Event<Tt, Ty>& Event<Tt, Ty>::operator=(const Event<Tt, Ty>& other){
    _name = other._name;
    _when = other._when;
    _check_if = other._check_if;
    _mask = other._mask;
    _hide_mask = other._hide_mask;
    _counter = other._counter;
    _args = other._args;
    _copy_event_data(other);
    return *this;
}

template<class Tt, class Ty>
void Event<Tt, Ty>::_set(const Tt& t, const Ty& q){
    delete _data;
    if (_mask != nullptr){
        _data = new EventData<Tt, Ty>(t, q, _mask(t, q, _args));
    }
    else{
        _data = new EventData<Tt, Ty>(t, q);
    }
    _counter++;
}

template<class Tt, class Ty>
void Event<Tt, Ty>::_copy_event_data(const Event<Tt, Ty>& other){
    delete _data;
    if (other._data != nullptr){
        _data = new EventData<Tt, Ty>;
        *_data = *other._data;
    }
    else{
        _data = nullptr;
    }
}

template<class Tt, class Ty>
void Event<Tt, Ty>::_clear(){
    delete _data;
    _data = nullptr;
}




//PreciseEvent CLASS

template<class Tt, class Ty>
PreciseEvent<Tt, Ty>& PreciseEvent<Tt, Ty>::operator=(const PreciseEvent<Tt, Ty>& other){
    if (&other != this){
        _event_tol = other._event_tol;
        Event<Tt, Ty>::operator=(other);
    }
    return *this;
}

template<class Tt, class Ty>
bool PreciseEvent<Tt, Ty>::determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) {
    this->_clear();
    Tt t_determined = t2;
    bool determined = false;
    if (this->check(t1, q1)){
        _ObjFun<Tt> f = [this, q](const Tt& t) ->Tt {
            return this->obj_fun(t, q(t));
        };
        Tt val1 = this->obj_fun(t1, q1);
        Tt val2 = this->obj_fun(t2, q2);
        if (val1 * val2 <= 0 && val1 != 0){
            t_determined = bisect(f, t1, t2, this->_event_tol)[2];
            determined = this->check(t_determined, q(t_determined));
        }
    }

    if (determined){
        this->_set(t_determined, q(t_determined));
    }
    return determined;

}



//PeriodicEvent CLASS


template<class Tt, class Ty>
PeriodicEvent<Tt, Ty>& PeriodicEvent<Tt, Ty>::operator=(const PeriodicEvent<Tt, Ty>& other){
    if (&other != this){
        _period = other._period;
        _start = other._start;
        _np = other._np;
        _np_previous = other._np_previous;
        _has_started = other._has_started;
        PreciseEvent<Tt, Ty>::operator=(other);
    }
    return *this;
}


template<class Tt, class Ty>
bool PeriodicEvent<Tt, Ty>::determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) {
    this->_clear();
    const int direction = (t2 > t1) ? 1 : -1;
    const Tt next = _has_started ? _start+(_np+direction)*_period : _start;
    if ( (t2*direction >= next*direction) && (next*direction > t1*direction) ){
        _np_previous = _np;
        _np += direction*this->_has_started;
        this->_set(next, q(next));
        this->_has_started = true;
        return true;
    }
    else{
        return false;
    }
}

template<class Tt, class Ty>
void PeriodicEvent<Tt, Ty>::go_back() {
    PreciseEvent<Tt, Ty>::go_back();
    if (_np != _np_previous){
        _np = _np_previous;
        if (_np_previous == 0){
            _has_started = false;
        }
    }
}




//RoughEvent CLASS



template<class Tt, class Ty>
RoughEvent<Tt, Ty>& RoughEvent<Tt, Ty>::operator=(const RoughEvent<Tt, Ty>& other){
    if (&other != this){
        Event<Tt, Ty>::operator=(other);
    }
    return *this;
}


template<class Tt, class Ty>
bool RoughEvent<Tt, Ty>::determine(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::function<Ty(const Tt&)>& q) {
    this->_clear();
    if (this->check(t1, q1) && this->check(t2, q2)){
        Tt val1 = this->obj_fun(t1, q1);
        Tt val2 = this->obj_fun(t2, q2);
        if (val1 * val2 <= 0 && val1 != 0){
            this->_set(t2, q2);
            return true;
        }
    }
    return false;
}

#endif