#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include <unordered_set>
#include "states.hpp"

template<typename T, int N>
using event_f = std::function<T(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<typename T, int N>
using is_event_f = std::function<bool(const T&, const vec<T, N>&, const std::vector<T>&)>;


template<typename T, int N>
class Event{

public:

    virtual bool                    determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) = 0;

    virtual bool                    is_stop_event() const;

    virtual bool                    is_leathal() const;

    virtual bool                    check(const T& t, const vec<T, N>& q) const;

    virtual void                    go_back();

    inline void                     set_args(const std::vector<T>& args);

    virtual Event<T, N>*            clone() const = 0;

    const ViewState<T, N>&          state() const;

    inline const std::string&       name() const;

    inline bool                     is_masked() const;

    inline const bool&              hides_mask() const;

    inline const size_t&            counter() const;

    virtual bool                    is_precise() const = 0;

    inline const std::vector<T>&    args() const;

    inline void                     remove();

    virtual                         ~Event() = default;


private:

    std::string         _name;
    event_f<T, N>       _when = nullptr;
    is_event_f<T, N>    _check_if = nullptr;
    Functor<T, N>       _mask = nullptr;
    bool                _hide_mask;
    size_t              _counter = 0;
    std::vector<T>      _args = {};
    ViewState<T, N>     _state;
    bool                _is_determined = false;

protected:

    Event(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if, Functor<T, N> mask, const bool& hide_mask);

    DEFAULT_RULE_OF_FOUR(Event);

    inline T        obj_fun(const T& t, const vec<T, N>& q) const;

    void            _set(const T& t, const vec<T, N>& q);

};

template<typename T, int N>
class PreciseEvent : public Event<T, N>{

public:

    PreciseEvent(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if=nullptr, Functor<T, N> mask=nullptr, const bool& hide_mask=false, const T& event_tol=1e-12): Event<T, N>(name, when, check_if, mask, hide_mask), _event_tol(event_tol){}

    DEFAULT_RULE_OF_FOUR(PreciseEvent);

    PreciseEvent<T, N>* clone() const override;

    bool                determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override;

    inline bool         is_precise() const final;

private:

    T _event_tol;

};


template<typename T, int N>
class PeriodicEvent : public PreciseEvent<T, N>{

public:

    PeriodicEvent(const std::string& name, const T& period, const T& start, Functor<T, N> mask=nullptr, const bool& hide_mask=false);

    PeriodicEvent(const std::string& name, const T& period, Functor<T, N> mask=nullptr, const bool& hide_mask=false);

    DEFAULT_RULE_OF_FOUR(PeriodicEvent);

    const T&                period() const;

    const T&                t_start() const;

    PeriodicEvent<T, N>*    clone() const override;

    bool                    determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override;

    void                    go_back() override;

    void                    set_start(const T& t);

protected:
    T _period;
    T _start;

private:
    long int _np = 0;
    long int _np_previous = 0;
    bool _has_started = false;

};


template<typename T, int N>
class RoughEvent : public Event<T, N>{

public:

    RoughEvent(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if=nullptr, Functor<T, N> mask=nullptr, const bool& hide_mask=false);

    DEFAULT_RULE_OF_FOUR(RoughEvent);

    RoughEvent<T, N>* clone() const override;

    bool              determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) final ;

    inline bool       is_precise() const final;

};











//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//----------------------------------IMPLEMENTATIONS--------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------










//Event CLASS

template<typename T, int N>
Event<T, N>::Event(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if, Functor<T, N> mask, const bool& hide_mask): _name(name), _when(when), _check_if(check_if), _mask(mask), _hide_mask(hide_mask && mask != nullptr){
    if (name == ""){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename T, int N>
inline T Event<T, N>::obj_fun(const T& t, const vec<T, N>& q) const{ 
    return _when(t, q, _args);
}

template<typename T, int N>
bool Event<T, N>::is_stop_event() const{
    return false;
}

template<typename T, int N>
bool Event<T, N>::is_leathal() const{
    return false;
}

template<typename T, int N>
bool Event<T, N>::check(const T& t, const vec<T, N>& q) const {
    return (_check_if == nullptr) ? true : _check_if(t, q, _args);
}

template<typename T, int N>
void Event<T, N>::set_args(const std::vector<T>& args){
    _args = args;
}

template<typename T, int N>
inline const std::string& Event<T, N>::name()const{
    return _name;
}

template<typename T, int N>
inline bool Event<T, N>::is_masked() const{
    return _mask != nullptr;
}

template<typename T, int N>
inline const bool& Event<T, N>::hides_mask() const{
    return _hide_mask;
}

template<typename T, int N>
inline const size_t& Event<T, N>::counter()const{
    return _counter;
}

template<typename T, int N>
inline const std::vector<T>& Event<T, N>::args()const{
    return _args;
}

template<typename T, int N>
inline void Event<T, N>::remove(){
    _is_determined = false;
}

template<typename T, int N>
void Event<T, N>::go_back(){
    if (_is_determined){
        _is_determined = false;
        _counter--;
    }
}

template<typename T, int N>
const ViewState<T, N>& Event<T, N>::state() const {
    if (!_is_determined){
        throw std::runtime_error("Event has not been determined");
    }
    return _state;
}

template<typename T, int N>
void Event<T, N>::_set(const T& t, const vec<T, N>& q){
    if (_mask != nullptr){
        _state.set(t, _mask(t, q, _args), q);
    }
    else{
        _state.set(t, q);
    }
    if (!_is_determined){
        _counter++;
        _is_determined = true;
    }
}



//PreciseEvent CLASS

template<typename T, int N>
bool PreciseEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->remove();
    T t_determined = t2;
    bool determined = false;
    if (this->check(t1, q1)){
        _ObjFun<T> f = [this, &q](const T& t) ->T {
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

template<typename T, int N>
PreciseEvent<T, N>* PreciseEvent<T, N>::clone() const{
    return new PreciseEvent<T, N>(*this);
}

template<typename T, int N>
inline bool PreciseEvent<T, N>::is_precise() const {
    return true;
}

//PeriodicEvent CLASS

template<typename T, int N>
PeriodicEvent<T, N>::PeriodicEvent(const std::string& name, const T& period, const T& start, Functor<T, N> mask, const bool& hide_mask): PreciseEvent<T, N>(name, nullptr, nullptr, mask, hide_mask, 0), _period(period), _start(start){
    if (period <= 0){
        throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
    }
}

template<typename T, int N>
PeriodicEvent<T, N>::PeriodicEvent(const std::string& name, const T& period, Functor<T, N> mask, const bool& hide_mask): PeriodicEvent<T, N>(name, period, inf<T>(), mask, hide_mask){}

template<typename T, int N>
const T& PeriodicEvent<T, N>::t_start() const{
    return _start;
}

template<typename T, int N>
const T& PeriodicEvent<T, N>::period() const{
    return _period;
}

template<typename T, int N>
PeriodicEvent<T, N>* PeriodicEvent<T, N>::clone() const{
    return new PeriodicEvent<T, N>(*this);
}

template<typename T, int N>
bool PeriodicEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->remove();
    const int direction = (t2 > t1) ? 1 : -1;
    const T next = _has_started ? _start+(_np+direction)*_period : _start;
    if ( (t2*direction >= next*direction) && ((next*direction > t1*direction) || (!_has_started && (next==t1))) ){
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

template<typename T, int N>
void PeriodicEvent<T, N>::go_back() {
    PreciseEvent<T, N>::go_back();
    if (_np != _np_previous){
        _np = _np_previous;
        if (_np_previous == 0){
            _has_started = false;
        }
    }
}

template<typename T, int N>
void PeriodicEvent<T, N>::set_start(const T& t) {
    if (abs(_start) == inf<T>()){
        _start = t;
    }
    else{
        throw std::runtime_error("Cannot reset starting point of PeriodicEvent");
    }
}


//RoughEvent CLASS

template<typename T, int N>
RoughEvent<T, N>::RoughEvent(const std::string& name, event_f<T, N> when, is_event_f<T, N> check_if, Functor<T, N> mask, const bool& hide_mask): Event<T, N>(name, when, check_if, mask, hide_mask){}

template<typename T, int N>
RoughEvent<T, N>* RoughEvent<T, N>::clone() const{
    return new RoughEvent<T, N>(*this);
}

template<typename T, int N>
inline bool RoughEvent<T, N>::is_precise() const {
    return false;
}

template<typename T, int N>
bool RoughEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) {
    this->remove();
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





template<typename T, int N>
class EventCollection{

public:

    EventCollection(std::initializer_list<Event<T, N>*> events){
        _copy(events.begin(), events.size());
    }

    EventCollection(std::initializer_list<const Event<T, N>*> events){
        _copy(events.begin(), events.size());
    }

    EventCollection(const std::vector<Event<T, N>*>& events){
        _copy(events.begin(), events.size());
    }

    EventCollection(const std::vector<const Event<T, N>*>& events){
        _copy(events.begin(), events.size());
    }

    EventCollection() = default;

    EventCollection(const EventCollection& other) : _events(other._events.size()), _names(other._names){
        for (size_t i=0; i<_events.size(); i++){
            _events[i] = other._events[i]->clone();
        }
    }

    EventCollection(EventCollection&& other) = default;

    ~EventCollection(){
        _clear();
    }

    EventCollection<T, N>& operator=(const EventCollection<T, N>& other){
        if (&other != this){
            _copy(other._events.begin(), other._events.size());
        }
        return *this;
    }

    EventCollection& operator=(EventCollection<T, N>&& other) = default;

    inline const Event<T, N>& operator[](const size_t& i) const {
        return *_events.at(i);
    }

    inline Event<T, N>& operator[](const size_t& i) {
        return *_events.at(i);
    }

    inline size_t size()const{
        return _events.size();
    }

    void set_args(const std::vector<T>& args){
        for (Event<T, N>* ev : _events) {
            ev->set_args(args);
        }
    }

    EventCollection<T, N> including(const Event<T, N>* event) const {
        if (_names.contains(event->name())) {
            throw std::runtime_error("Cannot include new Event because dublicate names are not allowed: " + event->name());
        }
        EventCollection<T, N> res(*this);
        res._events.push_back(event->clone());
        res._events.shrink_to_fit();
        return res;
    }

private:

    void _clear(){
        for (size_t i=0; i<_events.size(); i++){
            delete _events[i];
        }
        _events.clear();
    }

    template<class Iterator>
    void _copy(const Iterator& events, const size_t& size){

        // FIRST create a new vector with new allocated objects, because "events" might be
        // our current _events vector. We sort the vector to contain normal events first,
        // and stop_events after to improve runtime performance and not miss out on any stop_events
        // if a single step encounters multiple events.

        std::vector<Event<T, N>*> new_precise_events;
        std::vector<Event<T, N>*> new_rough_events;
        for (size_t i = 0; i < size; i++) {
            if (!_names.insert(events[i]->name()).second) {
                throw std::runtime_error("Duplicate Event name not allowed: " + events[i]->name());
            }
            if (events[i]->is_precise()) {
                new_precise_events.push_back(events[i]->clone());
            }
            else {
                new_rough_events.push_back(events[i]->clone());
            }
        }

        std::vector<Event<T, N>*> result(size);
        std::copy(new_precise_events.begin(), new_precise_events.end(), result.begin());
        std::copy(new_rough_events.begin(), new_rough_events.end(), result.begin() + new_precise_events.size());
        
        // NOW we can delete our current events
        _clear();

        _events = result;
    }

    std::vector<Event<T, N>*> _events;
    std::unordered_set<std::string> _names;

};




/*

(rule of 5)
In classes, explicitly declare:

protected copy/move constructors in pure virtual base classes

public copy/move constructors in concrete classes

similar for assignment and move assignment operator

public virtual destructor in base class

*/
#endif