#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include "tools.hpp"

template<class S>
using event_f = std::function<S(const S&, const Tensor<S>&, const std::vector<S>&)>;

template<class S>
using is_event_f = std::function<bool(const S&, const Tensor<S>&, const std::vector<S>&)>;

template<class S>
struct EventData{

    using Ty = Tensor<S>;

    const S& t() const {return _t;}
    const Ty& q() const {return _q;}
    const Ty& q_true() const {return _q_masked;}

    EventData(){};

    EventData(const S& t, const Ty& q):_t(t), _q(q), _q_masked(q), _masked(false){}

    EventData(const S& t, const Ty& q, const Ty& q_true):_t(t), _q(q), _q_masked(q_true), _masked(true){}

private:
    S _t;
    Ty _q;
    Ty _q_masked;
    bool _masked;
};


template<class S>
class Event{

using Ty = Tensor<S>;

public:

    virtual bool determine(const S& t1, const Ty& q1, const S& t2, const Ty& q2, const std::function<Ty(const S&)>& q) = 0;

    virtual bool is_stop_event() const {return false;}

    virtual bool is_leathal() const { return false;}

    virtual bool check(const S& t, const Ty& q) const {return (_check_if == nullptr) ? true : _check_if(t, q, _args);}

    virtual void go_back();

    inline void set_args(const std::vector<S>& args){ _args = args;}

    virtual Event<S>* clone() const = 0;

    const EventData<S>& data() const;

    inline const std::string& name()const{ return _name;}

    inline const bool& hide_mask()const{ return _hide_mask;}

    inline bool has_mask()const{ return _mask != nullptr;}

    virtual bool allows_checkpoint() const{ return !has_mask(); }

    inline const size_t& counter() const{ return _counter; }

    virtual bool is_precise() const = 0;

    virtual ~Event(){delete _data;}


private:

    std::string _name;
    event_f<S> _when = nullptr;
    is_event_f<S> _check_if = nullptr;
    Func<S> _mask = nullptr;
    bool _hide_mask; // variable that is only used externally for odesolvers to determine when and whether to call q_event or q_masked.
    size_t _counter = 0;
    std::vector<S> _args = {};
    EventData<S>* _data = nullptr;
    

protected:

    Event(const std::string& name, event_f<S> when, is_event_f<S> check_if, Func<S> mask, const bool& hide_mask): _name(name), _when(when), _check_if(check_if), _mask(mask), _hide_mask(hide_mask){
        if (name == ""){
            throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
        }
    }

    Event(const Event<S>& other): _name(other._name), _when(other._when), _check_if(other._check_if), _mask(other._mask), _hide_mask(other._hide_mask), _counter(other._counter), _args(other._args){
        _copy_event_data(other);
    }

    Event(Event<S>&& other): _name(std::move(other._name)), _when(std::move(other._when)), _check_if(std::move(other._check_if)), _mask(std::move(other._mask)), _hide_mask(other._hide_mask), _counter(other._counter), _args(std::move(other._args)), _data(other._data){other._data = nullptr;}

    Event<S>& operator=(const Event<S>& other);

    inline S obj_fun(const S& t, const Ty& q) const { return _when(t, q, _args); }

    void _set(const S& t, const Ty& q);

    void _copy_event_data(const Event<S>& other);

    void _clear();

};

template<class S>
class PreciseEvent : public Event<S>{

using Ty = Tensor<S>;

public:

    PreciseEvent(const std::string& name, event_f<S> when, is_event_f<S> check_if=nullptr, Func<S> mask=nullptr, const bool& hide_mask=false, const S& event_tol=1e-12): Event<S>(name, when, check_if, mask, hide_mask), _event_tol(event_tol){}

    PreciseEvent(const PreciseEvent<S>& other):Event<S>(other), _event_tol(other._event_tol){}

    PreciseEvent<S>& operator=(const PreciseEvent<S>& other);

    PreciseEvent<S>* clone() const override{
        return new PreciseEvent<S>(*this);
    }

    bool determine(const S& t1, const Ty& q1, const S& t2, const Ty& q2, const std::function<Ty(const S&)>& q) override;

    inline bool is_precise() const final { return true;}

private:

    S _event_tol;

};


template<class S>
class PeriodicEvent : public PreciseEvent<S>{

using Ty = Tensor<S>;

public:

    PeriodicEvent(const PeriodicEvent<S>& other):PreciseEvent<S>(other), _period(other._period), _start(other._start), _np(other._np), _np_previous(other._np_previous){}

    PeriodicEvent(const std::string& name, const S& period, const S& start=0, Func<S> mask=nullptr, const bool& hide_mask=false): PreciseEvent<S>(name, nullptr, nullptr, mask, hide_mask, 0), _period(period), _start(start){
        if (period <= 0){
            throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
        }
    }

    PeriodicEvent<S>& operator=(const PeriodicEvent<S>& other);

    PeriodicEvent<S>* clone() const override{
        return new PeriodicEvent<S>(*this);
    }

    bool determine(const S& t1, const Ty& q1, const S& t2, const Ty& q2, const std::function<Ty(const S&)>& q) override;

    void go_back() override;

private:
    S _period;
    S _start;
    long int _np = 0;
    long int _np_previous = 0;
    bool _has_started = false;

};


template<class S>
class RoughEvent : public Event<S>{

using Ty = Tensor<S>;

public:

    RoughEvent(const RoughEvent<S>& other) : Event<S>(other){}

    RoughEvent(const std::string& name, event_f<S> when, is_event_f<S> check_if=nullptr, Func<S> mask=nullptr, const bool& hide_mask=false): Event<S>(name, when, check_if, mask, hide_mask){}

    RoughEvent<S>& operator=(const RoughEvent<S>& other);

    RoughEvent<S>* clone() const override{
        return new RoughEvent<S>(*this);
    }

    bool determine(const S& t1, const Ty& q1, const S& t2, const Ty& q2, const std::function<Ty(const S&)>& q) final ;

    inline bool allows_checkpoint() const override {return false; }

    inline bool is_precise() const final { return false;}

};











//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//----------------------------------IMPLEMENTATIONS--------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------










//Event CLASS

template<class S>
void Event<S>::go_back(){
    if (_data != nullptr){
        _clear();
        _counter--;
    }
}

template<class S>
const EventData<S>& Event<S>::data() const {
    if (_data == nullptr){
        throw std::runtime_error("Event has not been determined");
    }
    return *_data;
}

template<class S>
Event<S>& Event<S>::operator=(const Event<S>& other){
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

template<class S>
void Event<S>::_set(const S& t, const Tensor<S>& q){
    delete _data;
    if (_mask != nullptr){
        _data = new EventData<S>(t, q, _mask(t, q, _args));
    }
    else{
        _data = new EventData<S>(t, q);
    }
    _counter++;
}

template<class S>
void Event<S>::_copy_event_data(const Event<S>& other){
    delete _data;
    if (other._data != nullptr){
        _data = new EventData<S>;
        *_data = *other._data;
    }
    else{
        _data = nullptr;
    }
}

template<class S>
void Event<S>::_clear(){
    delete _data;
    _data = nullptr;
}




//PreciseEvent CLASS

template<class S>
PreciseEvent<S>& PreciseEvent<S>::operator=(const PreciseEvent<S>& other){
    if (&other != this){
        _event_tol = other._event_tol;
        Event<S>::operator=(other);
    }
    return *this;
}

template<class S>
bool PreciseEvent<S>::determine(const S& t1, const Tensor<S>& q1, const S& t2, const Tensor<S>& q2, const std::function<Tensor<S>(const S&)>& q) {
    this->_clear();
    S t_determined = t2;
    bool determined = false;
    if (this->check(t1, q1)){
        _ObjFun<S> f = [this, q](const S& t) -> S {
            return this->obj_fun(t, q(t));
        };
        S val1 = this->obj_fun(t1, q1);
        S val2 = this->obj_fun(t2, q2);
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


template<class S>
PeriodicEvent<S>& PeriodicEvent<S>::operator=(const PeriodicEvent<S>& other){
    if (&other != this){
        _period = other._period;
        _start = other._start;
        _np = other._np;
        _np_previous = other._np_previous;
        _has_started = other._has_started;
        PreciseEvent<S>::operator=(other);
    }
    return *this;
}


template<class S>
bool PeriodicEvent<S>::determine(const S& t1, const Ty& q1, const S& t2, const Ty& q2, const std::function<Ty(const S&)>& q) {
    this->_clear();
    const int direction = (t2 > t1) ? 1 : -1;
    const S next = _has_started ? _start+(_np+direction)*_period : _start;
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

template<class S>
void PeriodicEvent<S>::go_back() {
    PreciseEvent<S>::go_back();
    if (_np != _np_previous){
        _np = _np_previous;
        if (_np_previous == 0){
            _has_started = false;
        }
    }
}




//RoughEvent CLASS



template<class S>
RoughEvent<S>& RoughEvent<S>::operator=(const RoughEvent<S>& other){
    if (&other != this){
        Event<S>::operator=(other);
    }
    return *this;
}


template<class S>
bool RoughEvent<S>::determine(const S& t1, const Tensor<S>& q1, const S& t2, const Tensor<S>& q2, const std::function<Tensor<S>(const S&)>& q) {
    this->_clear();
    if (this->check(t1, q1) && this->check(t2, q2)){
        S val1 = this->obj_fun(t1, q1);
        S val2 = this->obj_fun(t2, q2);
        if (val1 * val2 <= 0 && val1 != 0){
            this->_set(t2, q2);
            return true;
        }
    }
    return false;
}

#endif