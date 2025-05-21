#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include "tools.hpp"

template<class T, int N>
using event_f = std::function<T(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N>
using is_event_f = std::function<bool(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N>
struct EventData{

    const T& t() const {return _t;}
    const vec<T, N>& q() const {return _q;}
    const vec<T, N>& q_true() const {return _q_masked;}

    EventData(){};

    EventData(const T& t, const vec<T, N>& q):_t(t), _q(q), _q_masked(q), _masked(false){}

    EventData(const T& t, const vec<T, N>& q, const vec<T, N>& q_true):_t(t), _q(q), _q_masked(q_true), _masked(true){}

private:
    T _t;
    vec<T, N> _q;
    vec<T, N> _q_masked;
    bool _masked;
};


template<class T, int N>
class Event{

public:

    virtual bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) = 0;

    virtual bool is_stop_event() const {return false;}

    virtual bool is_leathal() const { return false;}

    virtual bool check(const T& t, const vec<T, N>& q) const {return (_check_if == nullptr) ? true : _check_if(t, q, _args);}

    virtual void go_back();

    inline void set_args(const std::vector<T>& args){ _args = args;}

    virtual Event<T, N>* clone() const = 0;

    const EventData<T, N>& data() const;

    inline const std::string& name()const{ return _name;}

    inline const bool& hide_mask()const{ return _hide_mask;}

    inline bool has_mask()const{ return _mask != nullptr;}

    virtual bool allows_checkpoint() const{ return !has_mask(); }

    inline const size_t& counter() const{ return _counter; }

    virtual bool is_precise() const = 0;

    virtual ~Event(){delete _data;}


private:

    std::string _name;
    event_f<T, N> _when = nullptr;
    is_event_f<T, N> _check_if = nullptr;
    Func<T, N> _mask = nullptr;
    bool _hide_mask; // variable that is only used externally for odesolvers to determine when and whether to call q_event or q_masked.
    size_t _counter = 0;
    std::vector<T> _args = {};
    EventData<T, N>* _data = nullptr;
    

protected:

    Event(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if, Func<T, N> mask, const bool& hide_mask): _name(name), _when(when), _check_if(check_if), _mask(mask), _hide_mask(hide_mask){
        if (name == ""){
            throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
        }
    }

    Event(const Event<T, N>& other): _name(other._name), _when(other._when), _check_if(other._check_if), _mask(other._mask), _hide_mask(other._hide_mask), _counter(other._counter), _args(other._args){
        _copy_event_data(other);
    }

    Event(Event<T, N>&& other): _name(std::move(other._name)), _when(std::move(other._when)), _check_if(std::move(other._check_if)), _mask(std::move(other._mask)), _hide_mask(other._hide_mask), _counter(other._counter), _args(std::move(other._args)), _data(other._data){other._data = nullptr;}

    Event<T, N>& operator=(const Event<T, N>& other);

    inline T obj_fun(const T& t, const vec<T, N>& q) const { return _when(t, q, _args); }

    void _set(const T& t, const vec<T, N>& q);

    void _copy_event_data(const Event<T, N>& other);

    void _clear();

};

template<class T, int N>
class PreciseEvent : public Event<T, N>{

public:

    PreciseEvent(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if=nullptr, Func<T, N> mask=nullptr, const bool& hide_mask=false, const T& event_tol=1e-12): Event<T, N>(name, when, check_if, mask, hide_mask), _event_tol(event_tol){}

    PreciseEvent(const PreciseEvent<T, N>& other):Event<T, N>(other), _event_tol(other._event_tol){}

    PreciseEvent<T, N>& operator=(const PreciseEvent<T, N>& other);

    PreciseEvent<T, N>* clone() const override{
        return new PreciseEvent<T, N>(*this);
    }

    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override;

    inline bool is_precise() const final { return true;}

private:

    T _event_tol;

};


template<class T, int N>
class PeriodicEvent : public PreciseEvent<T, N>{

public:

    PeriodicEvent(const PeriodicEvent<T, N>& other):PreciseEvent<T, N>(other), _period(other._period), _start(other._start), _np(other._np), _np_previous(other._np_previous){}

    PeriodicEvent(const std::string& name, const T& period, const T& start=0, Func<T, N> mask=nullptr, const bool& hide_mask=false): PreciseEvent<T, N>(name, nullptr, nullptr, mask, hide_mask, 0), _period(period), _start(start){
        if (period <= 0){
            throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
        }
    }

    PeriodicEvent<T, N>& operator=(const PeriodicEvent<T, N>& other);

    PeriodicEvent<T, N>* clone() const override{
        return new PeriodicEvent<T, N>(*this);
    }

    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override;

    void go_back() override;

private:
    T _period;
    T _start;
    long int _np = 0;
    long int _np_previous = 0;
    bool _has_started = false;

};


template<class T, int N>
class RoughEvent : public Event<T, N>{

public:

    RoughEvent(const RoughEvent<T, N>& other) : Event<T, N>(other){}

    RoughEvent(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if=nullptr, Func<T, N> mask=nullptr, const bool& hide_mask=false): Event<T, N>(name, when, check_if, mask, hide_mask){}

    RoughEvent<T, N>& operator=(const RoughEvent<T, N>& other);

    RoughEvent<T, N>* clone() const override{
        return new RoughEvent<T, N>(*this);
    }

    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) final ;

    inline bool allows_checkpoint() const override {return false; }

    inline bool is_precise() const final { return false;}

};











//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//----------------------------------IMPLEMENTATIONS--------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------










//Event CLASS

template<class T, int N>
void Event<T, N>::go_back(){
    if (_data != nullptr){
        _clear();
        _counter--;
    }
}

template<class T, int N>
const EventData<T, N>& Event<T, N>::data() const {
    if (_data == nullptr){
        throw std::runtime_error("Event has not been determined");
    }
    return *_data;
}

template<class T, int N>
Event<T, N>& Event<T, N>::operator=(const Event<T, N>& other){
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

template<class T, int N>
void Event<T, N>::_set(const T& t, const vec<T, N>& q){
    delete _data;
    if (_mask != nullptr){
        _data = new EventData<T, N>(t, q, _mask(t, q, _args));
    }
    else{
        _data = new EventData<T, N>(t, q);
    }
    _counter++;
}

template<class T, int N>
void Event<T, N>::_copy_event_data(const Event<T, N>& other){
    delete _data;
    if (other._data != nullptr){
        _data = new EventData<T, N>;
        *_data = *other._data;
    }
    else{
        _data = nullptr;
    }
}

template<class T, int N>
void Event<T, N>::_clear(){
    delete _data;
    _data = nullptr;
}




//PreciseEvent CLASS

template<class T, int N>
PreciseEvent<T, N>& PreciseEvent<T, N>::operator=(const PreciseEvent<T, N>& other){
    if (&other != this){
        _event_tol = other._event_tol;
        Event<T, N>::operator=(other);
    }
    return *this;
}

template<class T, int N>
bool PreciseEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->_clear();
    T t_determined = t2;
    bool determined = false;
    if (this->check(t1, q1)){
        _ObjFun<T> f = [this, q](const T& t) ->T {
            return this->obj_fun(t, q(t));
        };
        T val1 = this->obj_fun(t1, q1);
        T val2 = this->obj_fun(t2, q2);
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


template<class T, int N>
PeriodicEvent<T, N>& PeriodicEvent<T, N>::operator=(const PeriodicEvent<T, N>& other){
    if (&other != this){
        _period = other._period;
        _start = other._start;
        _np = other._np;
        _np_previous = other._np_previous;
        _has_started = other._has_started;
        PreciseEvent<T, N>::operator=(other);
    }
    return *this;
}


template<class T, int N>
bool PeriodicEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->_clear();
    const int direction = (t2 > t1) ? 1 : -1;
    const T next = _has_started ? _start+(_np+direction)*_period : _start;
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

template<class T, int N>
void PeriodicEvent<T, N>::go_back() {
    PreciseEvent<T, N>::go_back();
    if (_np != _np_previous){
        _np = _np_previous;
        if (_np_previous == 0){
            _has_started = false;
        }
    }
}




//RoughEvent CLASS



template<class T, int N>
RoughEvent<T, N>& RoughEvent<T, N>::operator=(const RoughEvent<T, N>& other){
    if (&other != this){
        Event<T, N>::operator=(other);
    }
    return *this;
}


template<class T, int N>
bool RoughEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->_clear();
    if (this->check(t1, q1) && this->check(t2, q2)){
        T val1 = this->obj_fun(t1, q1);
        T val2 = this->obj_fun(t2, q2);
        if (val1 * val2 <= 0 && val1 != 0){
            this->_set(t2, q2);
            return true;
        }
    }
    return false;
}

#endif