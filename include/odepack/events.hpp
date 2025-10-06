#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include <stdexcept>
#include <unordered_set>
#include "states.hpp"

template<typename T, int N>
struct _EventObjFun;

template<typename T, int N>
T event_obj_func(const T& t, const void* obj){
    const _EventObjFun<T, N>* ptr = reinterpret_cast<const _EventObjFun<T, N>*>(obj);
    ptr->local_interp(ptr->q, t, ptr->obj);
    return ptr->event->obj_fun(t, ptr->q);
}

template<typename T, int N>
class Event{

public:

    virtual bool                    determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) = 0;

    virtual bool                    is_stop_event() const;

    virtual bool                    is_leathal() const;

    virtual void                    go_back();

    virtual void                    reset();

    inline void                     set_args(const std::vector<T>& args);

    virtual Event<T, N>*            clone() const = 0;

    const ViewState<T, N>&          state() const;

    inline const std::string&       name() const;

    inline const int&               dir() const;

    inline bool                     is_masked() const;

    inline const bool&              hides_mask() const;

    inline const size_t&            counter() const;

    virtual bool                    is_precise() const = 0;

    inline const std::vector<T>&    args() const;

    inline void                     remove();

    inline T                        obj_fun(const T& t, const T* q) const;
    
    virtual                         ~Event() = default;


private:

    std::string         _name;
    ObjFun<T>           _when = nullptr;
    int                 _dir = 1;
    Func<T>             _mask = nullptr;
    bool                _hide_mask;
    size_t              _counter = 0;
    std::vector<T>      _args = {};
    ViewState<T, N>     _state;
    bool                _is_determined = false;

protected:

    const void*         _obj = nullptr;

    mutable vec<T, N>   _tmp;

    mutable vec<T, N>   _tmp_mask;

    Event(const std::string& name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, const void* obj);

    Event() = default;

    DEFAULT_RULE_OF_FOUR(Event);

    void                _set(const T& t, const vec<T, N>& q);

};

template<typename T, int N>
class PreciseEvent : public Event<T, N>{

public:

    PreciseEvent(const std::string& name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, T event_tol=1e-12, const void* obj = nullptr): Event<T, N>(name, when, dir, mask, hide_mask, obj), _event_tol(event_tol){}

    PreciseEvent() = default;

    DEFAULT_RULE_OF_FOUR(PreciseEvent);

    PreciseEvent<T, N>* clone() const override;

    bool                determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) override;

    inline bool         is_precise() const final;

private:

    T _event_tol;

};


template<typename T, int N>
class PeriodicEvent : public PreciseEvent<T, N>{

public:

    PeriodicEvent(const std::string& name, T period, T start, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent(const std::string& name, T period, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent() = default;

    DEFAULT_RULE_OF_FOUR(PeriodicEvent);

    const T&                period() const;

    const T&                t_start() const;

    PeriodicEvent<T, N>*    clone() const override;

    bool                    determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) override;

    void                    go_back() override;

    virtual void            set_start(const T& t);

    void                    reset() override;

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

    RoughEvent(const std::string& name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    RoughEvent() = default;

    DEFAULT_RULE_OF_FOUR(RoughEvent);

    RoughEvent<T, N>* clone() const override;

    bool              determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) final ;

    inline bool       is_precise() const final;

};



template<typename EventType, typename ObjType>
class ObjectOwningEvent : public EventType{

public:

    template<typename... Args>
    ObjectOwningEvent(const ObjType& obj, Args&&... args): EventType(std::forward<Args>(args)...), _object(obj) {
        this->_obj = &_object;
    }

    ObjectOwningEvent(const ObjectOwningEvent& other) : EventType(other), _object(other._object) {
        this->_obj = &_object;
    }

    ObjectOwningEvent(ObjectOwningEvent&& other) : EventType(std::move(other)), _object(std::move(other._object)) {
        this->_obj = &_object;
    }

    ObjectOwningEvent& operator=(const ObjectOwningEvent& other){
        if (&other != this){
            EventType::operator=(other);
            _object = other._object;
            this->_obj = &_object;
        }
        return *this;
    }

    ObjectOwningEvent& operator=(ObjectOwningEvent&& other){
        if (&other != this){
            EventType::operator=(std::move(other));
            _object = std::move(other._object);
            this->_obj = &_object;
        }
        return *this;
    }

    ObjectOwningEvent<EventType, ObjType>* clone() const override{
        return new ObjectOwningEvent<EventType, ObjType>(*this);
    }

protected:

    ObjType _object;

};








//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//----------------------------------IMPLEMENTATIONS--------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------










//Event CLASS

template<typename T, int N>
Event<T, N>::Event(const std::string& name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, const void* obj): _name(name), _when(when), _dir(sgn(dir)), _mask(mask), _hide_mask(hide_mask && mask != nullptr), _obj(obj){
    if (name == ""){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename T, int N>
inline T Event<T, N>::obj_fun(const T& t, const T* q) const{ 
    return _when(t, q, _args.data(), _obj);
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
void Event<T, N>::set_args(const std::vector<T>& args){
    _args = args;
}

template<typename T, int N>
inline const std::string& Event<T, N>::name()const{
    return _name;
}

template<typename T, int N>
inline const int& Event<T, N>::dir()const{
    return _dir;
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
void Event<T, N>::reset(){
    _counter = 0;
    _state = ViewState<T, N>();
    _is_determined = false;
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
        this->_tmp_mask.resize(q.size());
        _mask(this->_tmp_mask.data(), t, q.data(), _args.data(), _obj);
        _state.set(t, this->_tmp_mask, q);
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
bool PreciseEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) {
    this->remove();
    T t_determined = t2;
    bool determined = false;

    T val1 = this->obj_fun(t1, q1.data());
    T val2 = this->obj_fun(t2, q2.data());

    int t_dir = sgn(t2-t1);
    const int& d = this->dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && val1 != 0){
        this->_tmp.resize(q1.size());
        _EventObjFun<T, N> _f{obj, this, q, this->_tmp.data()};
        t_determined = bisect(event_obj_func<T, N>, t1, t2, this->_event_tol, &_f)[2];
        determined = true;
    }

    if (determined){
        q(this->_tmp.data(), t_determined, obj);
        this->_set(t_determined, this->_tmp);
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
PeriodicEvent<T, N>::PeriodicEvent(const std::string& name, T period, T start, Func<T> mask, bool hide_mask, const void* obj): PreciseEvent<T, N>(name, nullptr, 0, mask, hide_mask, 0, obj), _period(period), _start(start){
    if (period <= 0){
        throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
    }
}

template<typename T, int N>
PeriodicEvent<T, N>::PeriodicEvent(const std::string& name, T period, Func<T> mask, bool hide_mask, const void* obj): PeriodicEvent<T, N>(name, period, inf<T>(), mask, hide_mask, obj){}

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
bool PeriodicEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>&, FuncLike<T> q, const void* obj) {
    this->remove();
    const int direction = (t2 > t1) ? 1 : -1;
    const T next = _has_started ? _start+(_np+direction)*_period : _start;
    if ( (t2*direction >= next*direction) && ((next*direction > t1*direction) || (!_has_started && (next==t1))) ){
        _np_previous = _np;
        _np += direction*this->_has_started;
        this->_tmp.resize(q1.size());
        q(this->_tmp.data(), next, obj);
        this->_set(next, this->_tmp);
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

template<typename T, int N>
void PeriodicEvent<T, N>::reset(){
    Event<T, N>::reset();
    _np = 0;
    _np_previous = 0;
    _has_started = false;
}


//RoughEvent CLASS

template<typename T, int N>
RoughEvent<T, N>::RoughEvent(const std::string& name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, const void* obj): Event<T, N>(name, when, dir, mask, hide_mask, obj){}

template<typename T, int N>
RoughEvent<T, N>* RoughEvent<T, N>::clone() const{
    return new RoughEvent<T, N>(*this);
}

template<typename T, int N>
inline bool RoughEvent<T, N>::is_precise() const {
    return false;
}

template<typename T, int N>
bool RoughEvent<T, N>::determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T>, const void*) {
    this->remove();

    T val1 = this->obj_fun(t1, q1.data());
    T val2 = this->obj_fun(t2, q2.data());
    int t_dir = sgn(t2-t1);
    const int& d = this->dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && val1 != 0){
        this->_set(t2, q2);
        return true;
    }
    return false;
}





template<typename T, int N>
class EventCollection{

public:

    EventCollection(std::initializer_list<const Event<T, N>*> events){
        _copy(events.begin(), events.size());
    }

    template<typename EventIterator>
    EventCollection(const EventIterator& events){
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
            _clear();
            _names.clear();
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

    void reset(){
        for (size_t i=0; i<this->size(); i++){
            this->_events[i]->reset();
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

template<typename T, int N>
struct _EventObjFun{
    const void* obj;
    const Event<T, N>* event;
    FuncLike<T> local_interp;
    T* q;
};


#endif