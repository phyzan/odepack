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

template<typename T, size_t N>
struct _EventObjFun;

template<typename T, size_t N>
T event_obj_func(const T& t, const void* obj);

template<typename T, size_t N>
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

    bool                            determine(EventState<T, N>& result, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj);

    virtual void                    reset();

    virtual void                    go_back();

    virtual bool                    is_stop_event() const;

    virtual bool                    is_leathal() const;

    virtual bool                    locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const = 0;

    virtual Event<T, N>*            clone() const = 0;

    inline virtual bool             is_time_based() const = 0;

    virtual ~Event() = default;

protected:

    Event(std::string name, Func<T> mask, bool hide_mask, const void* obj = nullptr);

    Event() = default;

    DEFAULT_RULE_OF_FOUR(Event);

    virtual void _register_it(const EventState<T, N>& res, State<const T> before, State<const T> after);

private:

    std::string             _name;
    Func<T>                 _mask = nullptr;
    bool                    _hide_mask;
    std::vector<T>          _args = {};
    size_t                  _counter = 0;


protected:
    const void* _obj;
    mutable Array1D<T, N>       _q_aux;
};

template<typename T, size_t N>
class PreciseEvent : public Event<T, N>{

public:

    PreciseEvent(std::string name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, T event_tol=1e-12, const void* obj = nullptr);

    PreciseEvent() = default;

    DEFAULT_RULE_OF_FOUR(PreciseEvent);

    inline T obj_fun(const T& t, const T* q) const;

    inline const int&   dir() const;

    PreciseEvent<T, N>* clone() const override;

    bool                locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const override;

    inline bool         is_time_based() const final;

private:

    ObjFun<T>           _when = nullptr;
    int                 _dir = 1;
    T                   _event_tol;

};


template<typename T, size_t N>
class PeriodicEvent : public Event<T, N>{

public:

    PeriodicEvent(std::string name, T period, T start, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent(std::string name, T period, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    PeriodicEvent() = default;

    DEFAULT_RULE_OF_FOUR(PeriodicEvent);

    const T&                period() const;

    const T&                t_start() const;

    PeriodicEvent<T, N>*    clone() const override;

    virtual void            set_start(const T& t);

    bool                    locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const override;

    inline bool             is_time_based() const final;

protected:
    T _period;
    T _start;
    mutable long int _n_aux = 0;

};


template<typename T, size_t N>
class TmaxEvent : public Event<T, N>{

public:

    TmaxEvent();

    DEFAULT_RULE_OF_FOUR(TmaxEvent);

    inline void set_goal(T tmax);

    inline bool goal_is_set() const;

    TmaxEvent<T, N>* clone() const override;

    bool is_stop_event() const override;

    inline bool is_time_based() const final;

    void reset() override;

    bool locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const override;

    void go_back() override;

private:

    void _register_it(const EventState<T, N>& res, State<const T> before, State<const T> after) override;

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

template<typename T, size_t N>
bool discard_event(const Event<T, N>& event, const Event<T, N>& mark);

template<typename T, size_t N>
class EventCollection{

public:

    template<typename EventIterator>
    EventCollection(const EventIterator& events);

    EventCollection(std::initializer_list<const Event<T, N>*> events);

    EventCollection() = default;

    EventCollection(const EventCollection& other);

    EventCollection(EventCollection&& other) noexcept;

    ~EventCollection();

    EventCollection<T, N>& operator=(const EventCollection<T, N>& other);

    EventCollection<T, N>& operator=(EventCollection<T, N>&& other) noexcept;

    inline const Event<T, N>& event(size_t i) const;

    inline const EventState<T, N>& state(size_t i) const;

    inline size_t size()const;

    inline size_t detection_size() const;

    inline size_t detection_times() const;

    void set_tmax(T tmax);

    void set_array_size(size_t size);

    void detect_all_between(State<const T> before, State<const T> after, FuncLike<T> q, const void* obj);

    const size_t* begin() const;

    const size_t* end() const;

    inline bool next_result();

    inline void restart_iter();

    const Event<T, N>* canon_event() const;

    const EventState<T, N>* canon_state() const;

    void set_start(T t0, int dir);

    void set_args(const T* args, size_t size);

    void reset();

private:

    void _realloc(size_t events);

    void _clear();

    template<typename Iterator>
    void _clone_events(const Iterator& events);

    void _copy(const EventCollection<T, N>& other);

    bool _is_prioritized(size_t i, size_t j, int dir);

    Event<T, N>** _events = nullptr;
    std::unordered_set<std::string> _names;
    EventState<T, N>* _states = nullptr;
    size_t _N_tot=0;

    //member variables that concern event detection
    size_t _N_detect=0;
    size_t _Nt=0;
    size_t _canon_idx;
    size_t* _event_idx=nullptr;
    size_t* _event_idx_start=nullptr;

    //iteration variable
    size_t _iter=0;

};

template<typename T, size_t N>
struct _EventObjFun{
    const void* obj;
    const PreciseEvent<T, N>* event;
    FuncLike<T> local_interp;
    T* q;
};


// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

template<typename T, size_t N>
T event_obj_func(const T& t, const void* obj){
    const auto* ptr = reinterpret_cast<const _EventObjFun<T, N>*>(obj);
    ptr->local_interp(ptr->q, t, ptr->obj);
    return ptr->event->obj_fun(t, ptr->q);
}

// Event CLASS implementations
template<typename T, size_t N>
Event<T, N>::Event(std::string name, Func<T> mask, bool hide_mask, const void* obj): _name(std::move(name)), _mask(mask), _hide_mask(hide_mask && mask != nullptr), _obj(obj){
    if (_name.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename T, size_t N>
bool Event<T, N>::is_stop_event() const{
    return false;
}

template<typename T, size_t N>
bool Event<T, N>::is_leathal() const{
    return false;
}

template<typename T, size_t N>
void Event<T, N>::set_args(const T* args, size_t size){
    _args.resize(size);
    copy_array(_args.data(), args, size);
}

template<typename T, size_t N>
inline const std::string& Event<T, N>::name()const{
    return _name;
}

template<typename T, size_t N>
inline bool Event<T, N>::is_masked() const{
    return _mask != nullptr;
}

template<typename T, size_t N>
inline bool Event<T, N>::hides_mask() const{
    return _hide_mask;
}

template<typename T, size_t N>
inline const std::vector<T>& Event<T, N>::args()const{
    return _args;
}

template<typename T, size_t N>
inline bool Event<T, N>::is_pure_temporal() const{
    return this->is_time_based() && !this->is_masked();
}

template<typename T, size_t N>
inline size_t Event<T, N>::counter() const{
    return _counter;
}

template<typename T, size_t N>
inline void Event<T, N>::set_aux_array_size(size_t size) const{
    _q_aux.resize(size);
}

template<typename T, size_t N>
bool Event<T, N>::determine(EventState<T, N>& result, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) {
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

template<typename T, size_t N>
void Event<T, N>::reset(){
    _counter = 0;
}

template<typename T, size_t N>
void Event<T, N>::go_back(){
    if (_counter > 0) {_counter--;}
}

template<typename T, size_t N>
void Event<T, N>::_register_it(const EventState<T, N>& res, State<const T> before, State<const T> after){
    this->_counter++;
}

// PreciseEvent CLASS implementations
template<typename T, size_t N>
PreciseEvent<T, N>::PreciseEvent(std::string name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, T event_tol, const void* obj): Event<T, N>(name, mask, hide_mask, obj), _when(when), _dir(sgn(dir)), _event_tol(event_tol){}

template<typename T, size_t N>
inline T PreciseEvent<T, N>::obj_fun(const T& t, const T* q) const{
    return _when(t, q, this->args().data(), this->_obj);
}

template<typename T, size_t N>
bool PreciseEvent<T, N>::locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const {

    T val1 = this->obj_fun(before.t(), before.vector());
    T val2 = this->obj_fun(after.t(), after.vector());

    int t_dir = sgn(after.t()-before.t());
    const int& d = this->dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && val1 != 0){
        _EventObjFun<T, N> _f{obj, this, q, this->_q_aux.data()};
        event.set_t(bisect(event_obj_func<T, N>, before.t(), after.t(), this->_event_tol, &_f)[2]);
        return true;
    }
    return false;
}

template<typename T, size_t N>
inline const int& PreciseEvent<T, N>::dir()const{
    return _dir;
}

template<typename T, size_t N>
PreciseEvent<T, N>* PreciseEvent<T, N>::clone() const{
    return new PreciseEvent<T, N>(*this);
}

template<typename T, size_t N>
inline bool PreciseEvent<T, N>::is_time_based() const{
    return false;
}

// PeriodicEvent CLASS implementations
template<typename T, size_t N>
PeriodicEvent<T, N>::PeriodicEvent(std::string name, T period, T start, Func<T> mask, bool hide_mask, const void* obj): Event<T, N>(name, mask, hide_mask, obj), _period(period), _start(start){
    if (period <= 0){
        throw std::runtime_error("Period in PeriodicEvent must be positive. If integrating backwards, events are still counted.");
    }
}

template<typename T, size_t N>
PeriodicEvent<T, N>::PeriodicEvent(std::string name, T period, Func<T> mask, bool hide_mask, const void* obj): PeriodicEvent<T, N>(name, period, inf<T>(), mask, hide_mask, obj){}

template<typename T, size_t N>
const T& PeriodicEvent<T, N>::t_start() const{
    return _start;
}

template<typename T, size_t N>
const T& PeriodicEvent<T, N>::period() const{
    return _period;
}

template<typename T, size_t N>
PeriodicEvent<T, N>* PeriodicEvent<T, N>::clone() const{
    return new PeriodicEvent<T, N>(*this);
}

template<typename T, size_t N>
bool PeriodicEvent<T, N>::locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const {
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

template<typename T, size_t N>
void PeriodicEvent<T, N>::set_start(const T& t) {
    if (!is_finite(_start)){
        _start = t;
    }
    else{
        throw std::runtime_error("Cannot reset starting point of PeriodicEvent");
    }
}

template<typename T, size_t N>
inline bool PeriodicEvent<T, N>::is_time_based() const{
    return true;
}

// TmaxEvent CLASS implementations
template<typename T, size_t N>
TmaxEvent<T, N>::TmaxEvent() : Event<T, N>("t-goal", nullptr, false) {}

template<typename T, size_t N>
inline void TmaxEvent<T, N>::set_goal(T tmax){
    _t_goal = tmax;
}

template<typename T, size_t N>
inline bool TmaxEvent<T, N>::goal_is_set() const{
    return is_finite(_t_goal);
}

template<typename T, size_t N>
TmaxEvent<T, N>* TmaxEvent<T, N>::clone() const{
    return new TmaxEvent<T, N>(*this);
}

template<typename T, size_t N>
bool TmaxEvent<T, N>::is_stop_event() const{
    return true;
}

template<typename T, size_t N>
inline bool TmaxEvent<T, N>::is_time_based() const{
    return true;
}

template<typename T, size_t N>
void TmaxEvent<T, N>::reset(){
    Event<T, N>::reset();
    _t_goal = inf<T>();
}

template<typename T, size_t N>
bool TmaxEvent<T, N>::locate(EventState<T, N>& event, State<const T> before, State<const T> after, FuncLike<T> q, const void* obj) const {
    if (!goal_is_set()) {return false;}

    int direction = (before.t() < after.t()) ? 1 : -1;
    if ((before.t()*direction < _t_goal*direction) && (_t_goal*direction <= after.t()*direction )){
        event.set_t(_t_goal);
        return true;
    }
    return false;
}

template<typename T, size_t N>
void TmaxEvent<T, N>::go_back(){
    Event<T, N>::go_back();
    _t_goal = _t_goal_last;
}

template<typename T, size_t N>
void TmaxEvent<T, N>::_register_it(const EventState<T, N>& res, State<const T> before, State<const T> after){
    Event<T, N>::_register_it(res, before, after);
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
template<typename T, size_t N>
bool discard_event(const Event<T, N>& event, const Event<T, N>& mark){
    if (!mark.is_masked() || mark.hides_mask()){
        return event.is_masked();
    }
    return !event.is_pure_temporal(); //mark is masked and not hidden
}

// EventCollection implementations
template<typename T, size_t N>
template<typename EventIterator>
EventCollection<T, N>::EventCollection(const EventIterator& events) {
    _realloc(events.size());
    _clone_events(events.begin());
}

template<typename T, size_t N>
EventCollection<T, N>::EventCollection(std::initializer_list<const Event<T, N>*> events){
    _realloc(events.size());
    _clone_events(events.begin());
}

template<typename T, size_t N>
EventCollection<T, N>::EventCollection(const EventCollection& other){
    _copy(other);
}

template<typename T, size_t N>
EventCollection<T, N>::EventCollection(EventCollection&& other) noexcept: _events(other._events), _names(std::move(other._names)), _states(other._states), _N_tot(other._N_tot), _N_detect(other._N_detect), _Nt(other._Nt), _canon_idx(other._canon_idx), _event_idx(other._event_idx), _event_idx_start(other._event_idx_start), _iter(other._iter){
    other._events = nullptr;
    other._states = nullptr;
    other._event_idx = nullptr;
    other._event_idx_start = nullptr;
    other._N_tot = 0;
}

template<typename T, size_t N>
EventCollection<T, N>::~EventCollection(){
    _clear();
}

template<typename T, size_t N>
EventCollection<T, N>& EventCollection<T, N>::operator=(const EventCollection<T, N>& other){
    if (&other != this){
        _copy(other);
    }
    return *this;
}

template<typename T, size_t N>
EventCollection<T, N>& EventCollection<T, N>::operator=(EventCollection<T, N>&& other) noexcept {
    if (&other != this){
        _clear();
        _events = other._events;
        _names = std::move(other._names);
        _states = other._states;
        _N_tot = other._N_tot;
        _N_detect = other._N_detect;
        _Nt = other._Nt;
        _canon_idx = other._canon_idx;
        _event_idx = other._event_idx;
        _event_idx_start = other._event_idx_start;
        _iter = other._iter;

        other._events = nullptr;
        other._states = nullptr;
        other._event_idx = nullptr;
        other._event_idx_start = nullptr;
        other._N_tot = 0;
    }
    return *this;
}

template<typename T, size_t N>
inline const Event<T, N>& EventCollection<T, N>::event(size_t i) const{
    return *_events[i];
}

template<typename T, size_t N>
inline const EventState<T, N>& EventCollection<T, N>::state(size_t i) const{
    return _states[i];
}

template<typename T, size_t N>
inline size_t EventCollection<T, N>::size()const{
    return _N_tot;
}

template<typename T, size_t N>
inline size_t EventCollection<T, N>::detection_size() const{
    return _N_detect;
}

template<typename T, size_t N>
inline size_t EventCollection<T, N>::detection_times() const{
    return _Nt;
}

template<typename T, size_t N>
void EventCollection<T, N>::set_tmax(T tmax){
    if (this->size() > 0){
        if (auto* p = dynamic_cast<TmaxEvent<T, N>*>(_events[0])){
            p->set_goal(tmax);
        }
    }
}

template<typename T, size_t N>
void EventCollection<T, N>::set_array_size(size_t size) {
    for (size_t i=0; i<_N_tot; i++){
        _states[i].resize(size);
        _events[i]->set_aux_array_size(size);
    }
}

template<typename T, size_t N>
void EventCollection<T, N>::detect_all_between(State<const T> before, State<const T> after, FuncLike<T> q, const void* obj){
    if (_N_tot == 0){
        return;
    }

    //detect all events and save their indices in order of priority in _event_idx.
    const int dir = sgn(after.t()-before.t());
    _N_detect = 0; //this is how many events have been triggered, and is the next available index in _event_idx_start
    for (size_t i=0; i<this->size(); i++){
        EventState<T, N>& event = _states[i];
        Event<T, N>* event_obj = _events[i];
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
    size_t mark = _N_tot;
    size_t i=0;
    while (i<_N_detect){
        if ((mark!=_N_tot) && (state(_event_idx[i]).t() != state(_event_idx[mark]).t())){
            mark = _N_tot;
        }
        if ((mark==_N_tot) && !event(_event_idx[i]).is_pure_temporal()){
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
    _canon_idx = _N_tot;
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
        const_cast<Event<T,N>&>(event(_event_idx[j])).go_back();
    }
    _N_detect = tmp;
    _iter = 0;
}

template<typename T, size_t N>
const size_t* EventCollection<T, N>::begin() const{
    return (_iter < _Nt) ? _event_idx + _event_idx_start[_iter] : nullptr;
}

template<typename T, size_t N>
const size_t* EventCollection<T, N>::end() const{
    return (_iter < _Nt) ? _event_idx + _event_idx_start[_iter+1] : nullptr;
}

template<typename T, size_t N>
inline bool EventCollection<T, N>::next_result() {
    if (_iter+1 < _Nt){
        _iter++;
        return true;
    }
    else if (_iter+1 == _Nt){
        _iter = _Nt;
    }
    return false;
}

template<typename T, size_t N>
inline void EventCollection<T, N>::restart_iter(){
    _iter = 0;
}

template<typename T, size_t N>
const Event<T, N>* EventCollection<T, N>::canon_event() const{
    return _canon_idx == _N_tot ? nullptr : _events[_canon_idx];
}

template<typename T, size_t N>
const EventState<T, N>* EventCollection<T, N>::canon_state() const{
    return _canon_idx == _N_tot ? nullptr : _states+_canon_idx;
}

template<typename T, size_t N>
void EventCollection<T, N>::set_start(T t0, int dir){
    for (size_t i=0; i<this->size(); i++){
        if (auto* p = dynamic_cast<PeriodicEvent<T, N>*>(_events[i])){
            if (!is_finite(p->t_start())){
                p->set_start(t0+p->period()*dir);
            }
            else if (p->t_start() == t0){
                throw std::runtime_error("The starting time of a periodic event cannot be set at the initial time of the ode solver.");
            }
        }
    }
}

template<typename T, size_t N>
void EventCollection<T, N>::set_args(const T* args, size_t size){
    for (size_t i=0; i<this->size(); i++){
        _events[i]->set_args(args, size);
    }
}

template<typename T, size_t N>
void EventCollection<T, N>::reset(){
    size_t arr_size;
    if (this->_N_tot > 0){
        arr_size = _states[0].nsys();
    }

    for (size_t i=0; i<_N_tot; i++){
        _events[i]->reset();
        _states[i] = EventState<T, N>();
        _event_idx[i] = 0;
        _event_idx_start[i] = 0;
    }

    this->set_array_size(arr_size);
    _N_detect = 0;
    _Nt = 0;
    _canon_idx = _N_tot;
    _iter = 0;
    _event_idx_start[_N_tot] = 0;
}

template<typename T, size_t N>
void EventCollection<T, N>::_realloc(size_t events){
    _clear();
    _events = new Event<T, N>*[events];
    _states = new EventState<T, N>[events];
    _event_idx = new size_t[events];
    _event_idx_start = new size_t[events+1];
    _N_tot = events;
    _canon_idx = _N_tot;
}

template<typename T, size_t N>
void EventCollection<T, N>::_clear(){
    for (size_t i=0; i<_N_tot; i++){
        delete _events[i];
        _events[i] = nullptr;
    }
    delete[] _events;
    delete[] _states;
    delete[] _event_idx;
    delete[] _event_idx_start;
}

template<typename T, size_t N>
template<typename Iterator>
void EventCollection<T, N>::_clone_events(const Iterator& events){
    _names.clear();
    for (size_t i = 0; i < _N_tot; i++) {
        if (!_names.insert(events[i]->name()).second) {
            throw std::runtime_error("Duplicate Event name not allowed: " + events[i]->name());
        }
        _events[i] = events[i]->clone();
    }
}

template<typename T, size_t N>
void EventCollection<T, N>::_copy(const EventCollection<T, N>& other){
    //events must have been deleted, and all arrays must have been allocated
    if (other._N_tot != _N_tot){
        _realloc(other._N_tot);
        _N_tot = other._N_tot;
    }
    else{
        for (size_t i=0; i<_N_tot; i++){
            delete _events[i];
        }
    }

    _N_detect = other._N_detect;
    _Nt = other._Nt;
    _canon_idx = other._canon_idx;
    _iter = other._iter;
    _clone_events(other._events);
    copy_array(_states, other._states, _N_tot);
    copy_array(_event_idx, other._event_idx, _N_tot);
    if (_N_tot>0){
        copy_array(_event_idx_start, other._event_idx_start, _N_tot+1);
    }

}

template<typename T, size_t N>
bool EventCollection<T, N>::_is_prioritized(size_t i, size_t j, int dir){
    if (state(i).t()*dir < state(j).t()*dir){
        return true;
    }
    else if (state(i).t()==state(j).t()){
        return i < j;
    }
    return false;
}


#endif
