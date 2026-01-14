#ifndef EVENTS_HPP
#define EVENTS_HPP

/*

@brief Header file dealing with Event encouters during an ODE integration.

During an ode integration, we might want to save specific events that are encoutered. For instance, we might want to specify at exactly what time the function that is integrated reached a spacific value. The Event classes handle such events, and are responsible for determining them with a desired accuracy.


*/

#include <unordered_set>
#include "tools.hpp"


// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
struct _EventObjFun;

template<typename T>
T event_obj_func(const T& t, const void* obj);

template<typename T>
class Event{

    // The obj parameter in "determine" is passed inside any functions that are passed in an Event object.
    // These include the "q" function in determine, or others passed in the constructors that accept a const void* parameter.

public:

    inline void                     set_args(const T* args, size_t size);

    inline const std::string&       name() const;

    inline bool                     is_masked() const;

    inline bool                     hides_mask() const;

    inline bool                     is_pure_temporal() const;

    inline size_t                   counter() const;

    inline const std::vector<T>&    args() const;

    inline void                     set_aux_array_size(size_t size) const;

    bool                            determine(EventState<T>& result, State<T> before, State<T> after, FuncLike<T> q, const void* obj);

    virtual void                    reset();

    virtual void                    go_back();

    virtual bool                    is_stop_event() const;

    virtual bool                    is_leathal() const;

    virtual bool                    locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const = 0;

    virtual Event<T>*            clone() const = 0;

    inline virtual bool             is_time_based() const = 0;

    virtual ~Event() = default;

protected:

    Event(std::string name, Func<T> mask, bool hide_mask, const void* obj = nullptr);

    Event() = default;

    DEFAULT_RULE_OF_FOUR(Event);

    virtual void _register_it(const EventState<T>& res, State<T> before, State<T> after);

private:

    std::string             _name;
    Func<T>                 _mask = nullptr;
    bool                    _hide_mask;
    std::vector<T>          _args = {};
    size_t                  _counter = 0;


protected:
    const void* _obj;
    mutable Array1D<T>       _q_aux;
};

template<typename T>
class PreciseEvent : public Event<T>{

public:

    PreciseEvent(std::string name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, T event_tol=1e-12, const void* obj = nullptr);

    PreciseEvent() = default;

    DEFAULT_RULE_OF_FOUR(PreciseEvent);

    inline T obj_fun(const T& t, const T* q) const;

    inline const int&   dir() const;

    PreciseEvent<T>* clone() const override;

    bool                locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const override;

    inline bool         is_time_based() const final;

private:

    ObjFun<T>           _when = nullptr;
    int                 _dir = 1;
    T                   _event_tol;

};


template<typename T>
class PeriodicEvent : public Event<T>{

public:

    PeriodicEvent(std::string name, T period, T start, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent(std::string name, T period, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent() = default;

    DEFAULT_RULE_OF_FOUR(PeriodicEvent);

    const T&                period() const;

    const T&                t_start() const;

    PeriodicEvent<T>*    clone() const override;

    virtual void            set_start(const T& t);

    bool                    locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const override;

    inline bool             is_time_based() const final;

protected:
    T _period;
    T _start;
    mutable long int _n_aux = 0;

};


template<typename T>
class TmaxEvent : public Event<T>{

public:

    TmaxEvent();

    DEFAULT_RULE_OF_FOUR(TmaxEvent);

    inline void set_goal(T tmax);

    inline bool goal_is_set() const;

    TmaxEvent<T>* clone() const override;

    bool is_stop_event() const override;

    inline bool is_time_based() const final;

    void reset() override;

    bool locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const override;

    void go_back() override;

private:

    void _register_it(const EventState<T>& res, State<T> before, State<T> after) override;

    T _t_goal = inf<T>();
    T _t_goal_last = inf<T>();

};


template<typename EventType, typename ObjType>
class ObjectOwningEvent : public EventType{

public:

    template<typename... Args>
    ObjectOwningEvent(const ObjType& obj, Args&&... args);

    ObjectOwningEvent(const ObjectOwningEvent& other);

    ObjectOwningEvent(ObjectOwningEvent&& other) noexcept;

    ObjectOwningEvent& operator=(const ObjectOwningEvent& other);

    ObjectOwningEvent& operator=(ObjectOwningEvent&& other) noexcept;

    ObjectOwningEvent<EventType, ObjType>* clone() const override;

protected:

    ObjType _object;

};

template<typename T>
bool discard_event(const Event<T>& event, const Event<T>& mark);

template<typename T>
using AnyEvent = PolyWrapper<Event<T>>;

template<typename T>
class EventView : private View1D<size_t>{

    using Base = View1D<size_t>;

public:

    EventView(const AnyEvent<T>* events, const size_t* detection, size_t size) : Base(detection, size), event_data(events) {};

    template<std::integral Int>
    const Event<T>* operator[](Int i) const{
        return event_data[Base::operator[](i)].ptr();
    }

    size_t size() const{
        return Base::size();
    }

    const AnyEvent<T>* event_data;

};



template<typename T>
class EventCollection{

public:

    EventCollection(const Event<T>*const* events, size_t size);

    EventCollection(const std::vector<const Event<T>*>& events);

    EventCollection() = default;

    DEFAULT_RULE_OF_FOUR(EventCollection)

    ~EventCollection() = default;

    inline const Event<T>& event(size_t i) const;

    inline const EventState<T>& state(size_t i) const;

    inline size_t size()const;

    inline size_t detection_size() const;

    inline size_t detection_times() const;

    void set_tmax(T tmax);

    void set_array_size(size_t size);

    void detect_all_between(State<T> before, State<T> after, FuncLike<T> q, const void* obj);

    EventView<T> event_view() const;

    const size_t* begin() const;

    const size_t* end() const;

    inline bool next_result();

    inline void restart_iter();

    const Event<T>* canon_event() const;

    const EventState<T>* canon_state() const;

    void set_start(T t0, int dir);

    void set_args(const T* args, size_t size);

    void reset();

private:

    bool _is_prioritized(size_t i, size_t j, int dir);

    Array1D<AnyEvent<T>>    _events;
    Array1D<EventState<T>>  _states;
    Array1D<size_t>         _event_idx;
    Array1D<size_t>         _event_idx_start;

    //member variables that concern event detection
    size_t                  _canon_idx;
    size_t                  _N_detect=0;
    size_t                  _Nt=0;
    //iteration variable
    size_t                  _iter=0;

};

template<typename T>
struct _EventObjFun{
    const void* obj;
    const PreciseEvent<T>* event;
    FuncLike<T> local_interp;
    T* q;
};


// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

template<typename T>
T event_obj_func(const T& t, const void* obj){
    const auto* ptr = reinterpret_cast<const _EventObjFun<T>*>(obj);
    ptr->local_interp(ptr->q, t, ptr->obj);
    return ptr->event->obj_fun(t, ptr->q);
}

// Event CLASS implementations
template<typename T>
Event<T>::Event(std::string name, Func<T> mask, bool hide_mask, const void* obj): _name(std::move(name)), _mask(mask), _hide_mask(hide_mask && mask != nullptr), _obj(obj){
    if (_name.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename T>
bool Event<T>::is_stop_event() const{
    return false;
}

template<typename T>
bool Event<T>::is_leathal() const{
    return false;
}

template<typename T>
void Event<T>::set_args(const T* args, size_t size){
    _args.resize(size);
    copy_array(_args.data(), args, size);
}

template<typename T>
inline const std::string& Event<T>::name()const{
    return _name;
}

template<typename T>
inline bool Event<T>::is_masked() const{
    return _mask != nullptr;
}

template<typename T>
inline bool Event<T>::hides_mask() const{
    return _hide_mask;
}

template<typename T>
inline const std::vector<T>& Event<T>::args()const{
    return _args;
}

template<typename T>
inline bool Event<T>::is_pure_temporal() const{
    return this->is_time_based() && !this->is_masked();
}

template<typename T>
inline size_t Event<T>::counter() const{
    return _counter;
}

template<typename T>
inline void Event<T>::set_aux_array_size(size_t size) const{
    _q_aux.resize(size);
}

template<typename T>
bool Event<T>::determine(EventState<T>& result, State<T> before, State<T> after, FuncLike<T> q, const void* obj) {
    if (locate(result, before, after, q, obj)){
        //result.t has been set, and thats all
        result.set_stepsize(after.habs());
        T* q_event = (this->_mask == nullptr) ? result.true_vector() : result.exposed_vector();
        q(q_event, result.t(), obj); //q_event has been set
        if (this->_mask != nullptr){
            copy_array(result.true_vector(), result.exposed_vector(), result.nsys());
            this->_mask(result.true_vector(), result.t(), result.exposed_vector(), _args.data(), _obj);
            result.choose_true = !this->hides_mask();
        }
        else{
            result.choose_true = true;
        }
        result.triggered = true;
        _register_it(result, before, after);
        return true;
    }
    result.triggered = false;
    return false;
}

template<typename T>
void Event<T>::reset(){
    _counter = 0;
}

template<typename T>
void Event<T>::go_back(){
    if (_counter > 0) {_counter--;}
}

template<typename T>
void Event<T>::_register_it(const EventState<T>& res, State<T> before, State<T> after){
    this->_counter++;
}

// PreciseEvent CLASS implementations
template<typename T>
PreciseEvent<T>::PreciseEvent(std::string name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, T event_tol, const void* obj): Event<T>(name, mask, hide_mask, obj), _when(when), _dir(sgn(dir)), _event_tol(event_tol){}

template<typename T>
inline T PreciseEvent<T>::obj_fun(const T& t, const T* q) const{
    return _when(t, q, this->args().data(), this->_obj);
}

template<typename T>
bool PreciseEvent<T>::locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const {

    T val1 = this->obj_fun(before.t(), before.vector());
    T val2 = this->obj_fun(after.t(), after.vector());

    int t_dir = sgn(before.t(), after.t());
    const int& d = this->dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && val1 != 0){
        _EventObjFun<T> _f{obj, this, q, this->_q_aux.data()};
        event.set_t(bisect(event_obj_func<T>, before.t(), after.t(), this->_event_tol, &_f)[2]);
        return true;
    }
    return false;
}

template<typename T>
inline const int& PreciseEvent<T>::dir()const{
    return _dir;
}

template<typename T>
PreciseEvent<T>* PreciseEvent<T>::clone() const{
    return new PreciseEvent<T>(*this);
}

template<typename T>
inline bool PreciseEvent<T>::is_time_based() const{
    return false;
}

// PeriodicEvent CLASS implementations
template<typename T>
PeriodicEvent<T>::PeriodicEvent(std::string name, T period, T start, Func<T> mask, bool hide_mask, const void* obj): Event<T>(name, mask, hide_mask, obj), _period(period), _start(start){
    if (period <= 0){
        throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
    }
}

template<typename T>
PeriodicEvent<T>::PeriodicEvent(std::string name, T period, Func<T> mask, bool hide_mask, const void* obj): PeriodicEvent<T>(name, period, inf<T>(), mask, hide_mask, obj){}

template<typename T>
const T& PeriodicEvent<T>::t_start() const{
    return _start;
}

template<typename T>
const T& PeriodicEvent<T>::period() const{
    return _period;
}

template<typename T>
PeriodicEvent<T>* PeriodicEvent<T>::clone() const{
    return new PeriodicEvent<T>(*this);
}

template<typename T>
bool PeriodicEvent<T>::locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const {
    if (!is_finite(_start)){
        return false;
    }

    int dir = sgn(before.t(), after.t());
    long int n = _n_aux;
    if ((n*_period+dir*_start) <= dir*before.t()){
        while (((++n)*_period+dir*_start) <= dir*before.t()){}
        _n_aux = n;
    }
    else{
        while (n*_period + dir*_start > dir*before.t()){
            _n_aux = n;
            n--;
        }
    }

    if ((_n_aux*_period+dir*_start) <= dir*after.t()){
        event.set_t(dir*_n_aux*_period+_start);
        return true;
    }
    return false;
}

template<typename T>
void PeriodicEvent<T>::set_start(const T& t) {
    if (!is_finite(_start)){
        _start = t;
    }
    else{
        throw std::runtime_error("Cannot reset starting point of PeriodicEvent");
    }
}

template<typename T>
inline bool PeriodicEvent<T>::is_time_based() const{
    return true;
}

// TmaxEvent CLASS implementations
template<typename T>
TmaxEvent<T>::TmaxEvent() : Event<T>("t-goal", nullptr, false) {}

template<typename T>
inline void TmaxEvent<T>::set_goal(T tmax){
    _t_goal = tmax;
}

template<typename T>
inline bool TmaxEvent<T>::goal_is_set() const{
    return is_finite(_t_goal);
}

template<typename T>
TmaxEvent<T>* TmaxEvent<T>::clone() const{
    return new TmaxEvent<T>(*this);
}

template<typename T>
bool TmaxEvent<T>::is_stop_event() const{
    return true;
}

template<typename T>
inline bool TmaxEvent<T>::is_time_based() const{
    return true;
}

template<typename T>
void TmaxEvent<T>::reset(){
    Event<T>::reset();
    _t_goal = inf<T>();
}

template<typename T>
bool TmaxEvent<T>::locate(EventState<T>& event, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const {
    if (!goal_is_set()) {return false;}

    int direction = (before.t() < after.t()) ? 1 : -1;
    if ((before.t()*direction < _t_goal*direction) && (_t_goal*direction <= after.t()*direction )){
        event.set_t(_t_goal);
        return true;
    }
    return false;
}

template<typename T>
void TmaxEvent<T>::go_back(){
    Event<T>::go_back();
    _t_goal = _t_goal_last;
}

template<typename T>
void TmaxEvent<T>::_register_it(const EventState<T>& res, State<T> before, State<T> after){
    Event<T>::_register_it(res, before, after);
    _t_goal_last = _t_goal;
    _t_goal = inf<T>();
}

// ObjectOwningEvent implementations
template<typename EventType, typename ObjType>
template<typename... Args>
ObjectOwningEvent<EventType, ObjType>::ObjectOwningEvent(const ObjType& obj, Args&&... args): EventType(std::forward<Args>(args)...), _object(obj) {
    this->_obj = &_object;
}

template<typename EventType, typename ObjType>
ObjectOwningEvent<EventType, ObjType>::ObjectOwningEvent(const ObjectOwningEvent& other) : EventType(other), _object(other._object) {
    this->_obj = &_object;
}

template<typename EventType, typename ObjType>
ObjectOwningEvent<EventType, ObjType>::ObjectOwningEvent(ObjectOwningEvent&& other) noexcept : EventType(std::move(other)), _object(std::move(other._object)) {
    this->_obj = &_object;
}

template<typename EventType, typename ObjType>
ObjectOwningEvent<EventType, ObjType>& ObjectOwningEvent<EventType, ObjType>::operator=(const ObjectOwningEvent& other){
    if (&other != this){
        EventType::operator=(other);
        _object = other._object;
        this->_obj = &_object;
    }
    return *this;
}

template<typename EventType, typename ObjType>
ObjectOwningEvent<EventType, ObjType>& ObjectOwningEvent<EventType, ObjType>::operator=(ObjectOwningEvent&& other) noexcept{
    if (&other != this){
        EventType::operator=(std::move(other));
        _object = std::move(other._object);
        this->_obj = &_object;
    }
    return *this;
}

template<typename EventType, typename ObjType>
ObjectOwningEvent<EventType, ObjType>* ObjectOwningEvent<EventType, ObjType>::clone() const{
    return new ObjectOwningEvent<EventType, ObjType>(*this);
}

// discard_event function
template<typename T>
bool discard_event(const Event<T>& event, const Event<T>& mark){
    if (!mark.is_masked() || mark.hides_mask()){
        return event.is_masked();
    }
    return !event.is_pure_temporal(); //mark is masked and not hidden
}

// EventCollection implementations
template<typename T>
EventCollection<T>::EventCollection(const std::vector<const Event<T>*>& events) : EventCollection(events.data(), events.size()) {}

template<typename T>
EventCollection<T>::EventCollection(const Event<T>*const* events, size_t size) : _events(size), _states(size), _event_idx(size), _event_idx_start(size+1), _canon_idx(size) {
    if (size == 0){return;}

    for (size_t i=0; i<size-1; i++){
        for (size_t j=i+1; j<size; j++){
            if (events[i]->name() == events[j]->name()){
                throw std::runtime_error("Duplicate Event name not allowed: " + events[i]->name());
            }
        }
    }

    for (size_t i=0; i<size; i++){
        _events[i].own(events[i]->clone());
    }
}

template<typename T>
inline const Event<T>& EventCollection<T>::event(size_t i) const{
    return *_events[i].ptr();
}

template<typename T>
inline const EventState<T>& EventCollection<T>::state(size_t i) const{
    return _states[i];
}

template<typename T>
inline size_t EventCollection<T>::size()const{
    return _events.size();
}

template<typename T>
inline size_t EventCollection<T>::detection_size() const{
    return _N_detect;
}

template<typename T>
inline size_t EventCollection<T>::detection_times() const{
    return _Nt;
}

template<typename T>
void EventCollection<T>::set_tmax(T tmax){
    if (this->size() > 0){
        if (TmaxEvent<T>* p = _events[0].template cast<TmaxEvent<T>>()){
            p->set_goal(tmax);
        }
    }
}

template<typename T>
void EventCollection<T>::set_array_size(size_t size) {
    for (size_t i=0; i< this->size(); i++){
        _states[i].resize(size);
        _events[i] ->set_aux_array_size(size);
    }
}

template<typename T>
void EventCollection<T>::detect_all_between(State<T> before, State<T> after, FuncLike<T> q, const void* obj){
    if (this->size() == 0){
        return;
    }

    //detect all events and save their indices in order of priority in _event_idx.
    const int dir = sgn(before.t(), after.t());
    _N_detect = 0; //this is how many events have been triggered, and is the next available index in _event_idx_start
    for (size_t i=0; i<this->size(); i++){
        EventState<T>& event = _states[i];
        Event<T>* event_obj = _events[i].ptr();
        if (event_obj->determine(event, before, after, q, obj)){
            long int j=static_cast<long int>(_N_detect)-1;
            while (j>=0 && _is_prioritized(i, j, dir)){
                _event_idx[j+1] = _event_idx[j];
                j--;
            }
            _event_idx[j+1] = i;
            _N_detect++;
        }
    }

    //discard "bad" events after the event that determines the main state vector
    size_t mark = this->size();
    size_t i=0;
    while (i<_N_detect){
        if ((mark!=this->size()) && (state(_event_idx[i]).t() != state(_event_idx[mark]).t())){
            mark = this->size();
        }
        if ((mark==this->size()) && !event(_event_idx[i]).is_pure_temporal()){
            mark = i;
        }
        else if (!event(_event_idx[i]).is_pure_temporal()){
            if (discard_event(event(_event_idx[i]), event(_event_idx[mark]))){
                _events[_event_idx[i]]->go_back();
                for (size_t j=i; j<_N_detect-1; j++){
                    _event_idx[j] = _event_idx[j+1];
                }
                _N_detect--;
                continue;
            }
        }
        i++;
    }

    //split the rest of events in groups of identical detection times and find canon (masked) event
    i=0;
    _canon_idx = this->size();
    _Nt = (_N_detect>0 ? 1 : 0);
    _event_idx_start[0] = 0;
    while (i<_N_detect){
        if (i>0 && (state(_event_idx[i]).t() != state(_event_idx[i-1]).t())){
            _event_idx_start[_Nt] = i;
            _Nt++;
        }
        if (event(_event_idx[i]).is_masked()){
            _canon_idx = _event_idx[i];
            i++;
            break;
        }
        i++;
    }

    //reverse the rest of events after the canon event detection time
    size_t j=i;
    while (j<_N_detect && (state(_event_idx[j]).t()== state(_event_idx[i-1]).t())){
        j++;
    }
    size_t tmp = j;
    _event_idx_start[_Nt] = j;
    for (; j<_N_detect; j++){
        const_cast<Event<T>&>(event(_event_idx[j])).go_back();
    }
    _N_detect = tmp;
    _iter = 0;
}

template<typename T>
EventView<T> EventCollection<T>::event_view() const{
    return (_iter < _Nt) ? EventView<T>(_events.data(), _event_idx.data() + _event_idx_start[_iter], this->detection_size()) : EventView<T>(nullptr, nullptr, 0);
}

template<typename T>
const size_t* EventCollection<T>::begin() const{
    return (_iter < _Nt) ? _event_idx.data() + _event_idx_start[_iter] : nullptr;
}

template<typename T>
const size_t* EventCollection<T>::end() const{
    return (_iter < _Nt) ? _event_idx.data() + _event_idx_start[_iter+1] : nullptr;
}

template<typename T>
inline bool EventCollection<T>::next_result() {
    if (_iter+1 < _Nt){
        _iter++;
        return true;
    }
    else if (_iter+1 == _Nt){
        _iter = _Nt;
    }
    return false;
}

template<typename T>
inline void EventCollection<T>::restart_iter(){
    _iter = 0;
}

template<typename T>
const Event<T>* EventCollection<T>::canon_event() const{
    return _canon_idx == this->size() ? nullptr : _events[_canon_idx].ptr();
}

template<typename T>
const EventState<T>* EventCollection<T>::canon_state() const{
    return _canon_idx == this->size() ? nullptr : _states.data()+_canon_idx;
}

template<typename T>
void EventCollection<T>::set_start(T t0, int dir){
    for (size_t i=0; i<this->size(); i++){
        if (PeriodicEvent<T>* p = _events[i].template cast<PeriodicEvent<T>>()){
            if (!is_finite(p->t_start())){
                p->set_start(t0+p->period()*dir);
            }
            else if (p->t_start() == t0){
                throw std::runtime_error("The starting time of a periodic event cannot be set at the initial time of the ode solver.");
            }
        }
    }
}

template<typename T>
void EventCollection<T>::set_args(const T* args, size_t size){
    for (size_t i=0; i<this->size(); i++){
        _events[i]->set_args(args, size);
    }
}

template<typename T>
void EventCollection<T>::reset(){
    size_t arr_size;
    if (this->size() > 0){
        arr_size = _states[0].nsys();
    }

    for (size_t i=0; i<this->size(); i++){
        _events[i]->reset();
        _states[i] = EventState<T>();
        _event_idx[i] = 0;
        _event_idx_start[i] = 0;
    }

    this->set_array_size(arr_size);
    _N_detect = 0;
    _Nt = 0;
    _canon_idx = this->size();
    _iter = 0;
    _event_idx_start[this->size()] = 0;
}



template<typename T>
bool EventCollection<T>::_is_prioritized(size_t i, size_t j, int dir){
    if (state(i).t()*dir < state(j).t()*dir){
        return true;
    }
    else if (state(i).t()==state(j).t()){
        return i < j;
    }
    return false;
}


#endif
