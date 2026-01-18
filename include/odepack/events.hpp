#ifndef EVENTS_HPP
#define EVENTS_HPP

#include "tools.hpp"

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
class Event{

public:

    virtual ~Event() = default;

    // ACCESSORS

    virtual const std::string&      name() const = 0;

    virtual bool                    is_masked() const = 0;

    virtual bool                    hides_mask() const = 0;

    virtual void                    apply_mask(T* out, const T& t, const T* q) const = 0;

    virtual bool                    is_pure_temporal() const = 0;

    virtual size_t                  counter() const = 0;

    virtual const std::vector<T>&   args() const = 0;

    virtual Event<T>*               clone() const = 0;

    virtual bool                    is_temporal() const = 0;

    virtual bool                    is_stop_event() const = 0;

    virtual bool                    is_lethal() const = 0;

    virtual int                     direction() const = 0;

    virtual bool                    is_located() const = 0;

    virtual const EventState<T>*    state() const = 0;

    // MODIFIERS

    virtual void setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction) = 0;

    virtual void set_args(const T* args, size_t size) = 0;

    virtual void deactivate() = 0;

    virtual bool locate(State<T> before, State<T> after, FuncLike<T> q, const void* obj) = 0;

    virtual bool register_it() = 0;

    virtual void reset() = 0;
};


template<typename Derived, typename T>
class EventBase : public Event<T>{

    // The obj parameter in "determine" is passed inside any functions that are passed in an Event object.
    // These include the "q" function in determine, or others passed in the constructors that accept a const void* parameter.

public:

    // ACCESSORS
    inline const std::string&       name() const;
    inline bool                     is_masked() const;
    inline bool                     hides_mask() const;
    inline void                     apply_mask(T* out, const T& t, const T* q) const;
    inline bool                     is_pure_temporal() const;
    inline size_t                   Nsys() const;
    inline size_t                   counter() const;
    inline const std::vector<T>&    args() const;
    Event<T>*                       clone() const;
    inline bool                     is_temporal() const;
    bool                            is_stop_event() const;
    bool                            is_lethal() const;
    inline int                      direction() const;
    inline bool                     is_located() const;
    inline const T&                 t_start() const;
    inline const EventState<T>*     state() const;

    // MODIFIERS
    void                            set_args(const T* args, size_t size);
    inline void                     setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);
    void                            deactivate();
    bool                            locate(State<T> before, State<T> after, FuncLike<T> q, const void* obj);
    bool                            register_it();
    void                            reset();

protected:

    using Main = EventBase<Derived, T>; // this class

    EventBase(std::string name, Func<T> mask, bool hide_mask, const void* obj = nullptr);
    EventBase() = default;
    DEFAULT_RULE_OF_FOUR(EventBase);

    // ================ STATIC OVERRIDE ======================
    inline bool locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    // ============= STATIC OVERRIDE (OPTIONAL) ================
    inline void register_impl();
    inline void reset_impl();
    inline void mask_impl(T* out, const T& t, const T* q) const;
    inline bool is_masked_impl() const;
    inline bool is_temporal_impl() const;
    inline bool stop_impl() const;
    inline bool kill_impl() const;
    //==========================================================

private:
    std::string             _name;
    std::vector<T>          _args = {};
    EventState<T>           _state;
    T                       _start = inf<T>();
protected:
    mutable Array1D<T>      _q_aux;
    const void*             _obj = nullptr;
private:
    size_t                  _counter = 0;
    Func<T>                 _mask = nullptr;
    int                     _direction;
    bool                    _hide_mask = false;
    bool                    _is_setup = false;
    bool                    _is_located = false;

};


template<typename T, typename Derived = void>
class PreciseEvent : public EventBase<GetDerived<PreciseEvent<T, Derived>, Derived>, T>{

    using Base = EventBase<GetDerived<PreciseEvent<T, Derived>, Derived>, T>;
    friend Base;

public:

    PreciseEvent(std::string name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, T event_tol=1e-12, const void* obj = nullptr);

    inline T    obj_fun(const T& t, const T* q) const;
    
    inline int  sign_change_dir() const;

protected:

    inline bool locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    ObjFun<T>           _when = nullptr;
    int                 _sign_dir = 1;
    T                   _event_tol;

};


template<typename T, typename Derived = void>
class PeriodicEvent : public EventBase<GetDerived<PeriodicEvent<T, Derived>, Derived>, T>{

    using Base = EventBase<GetDerived<PeriodicEvent<T, Derived>, Derived>, T>;
    friend Base;

public:

    PeriodicEvent(std::string name, T period, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    inline const T& period() const;

    inline size_t   np() const;

    inline T        delta_t_abs() const;

    inline T        delta_t() const;

protected:

    inline bool     is_temporal_impl() const;

    inline T        get_t(size_t n) const;

    inline bool     locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    inline void     register_impl();

    inline void     reset_impl();

private:

    T _period;
    size_t _n = 0;
    mutable size_t _n_aux = 0;

};

template<typename T, typename Derived = void>
class TmaxEvent : public EventBase<GetDerived<TmaxEvent<T, Derived>, Derived>, T>{

    using Base = EventBase<GetDerived<TmaxEvent<T, Derived>, Derived>, T>;
    friend Base;

public:

    TmaxEvent() = default;

    inline void set_goal(T tmax);

    inline bool goal_is_set() const;


protected:

    inline bool locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    inline void register_impl();

    inline void reset_impl();

    inline bool stop_impl() const;

    inline bool is_temporal_impl() const;

private:

    T _t_goal = inf<T>();

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

    EventView(const AnyEvent<T>* events, const size_t* detection, size_t size);

    template<std::integral Int>
    const Event<T>* operator[](Int i) const;

    size_t size() const;

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

    inline const Event<T>&      event(size_t i) const;
    inline const EventState<T>& state(size_t i) const;
    inline size_t               size() const;
    inline size_t               detection_size() const;
    inline size_t               detection_times() const;
    void                        set_tmax(T tmax);
    void                        setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);
    void                        set_args(const T* args, size_t size);
    void                        detect_all_between(State<T> before, State<T> after, FuncLike<T> q, const void* obj);
    EventView<T>                event_view() const;
    const size_t*               begin() const;
    const size_t*               end() const;
    inline bool                 next_result();
    inline void                 restart_iter();
    const Event<T>*             canon_event() const;
    const EventState<T>*        canon_state() const;
    void                        reset();

private:

    bool _is_prioritized(size_t i, size_t j, int dir);

    Array1D<AnyEvent<T>>    _events;
    Array1D<size_t>         _event_idx;
    Array1D<size_t>         _event_idx_start;

    //member variables that concern event detection
    size_t                  _canon_idx;
    size_t                  _N_detect=0;
    size_t                  _Nt=0;
    //iteration variable
    size_t                  _iter=0;

};

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

// EventBase implementations

template<typename Derived, typename T>
EventBase<Derived, T>::EventBase(std::string name, Func<T> mask, bool hide_mask, const void* obj) : _name(std::move(name)), _obj(obj), _mask(mask), _hide_mask(hide_mask) {
    if (_name.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename Derived, typename T>
inline const std::string& EventBase<Derived, T>::name() const{
    return _name;
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_masked() const{
    return THIS_C->is_masked_impl();
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::hides_mask() const{
    return _hide_mask && this->is_masked();
}

template<typename Derived, typename T>
inline void EventBase<Derived, T>::apply_mask(T* out, const T& t, const T* q) const{
    return THIS_C->mask_impl(out, t, q);
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_pure_temporal() const{
    return this->is_temporal() && !this->is_masked();
}

template<typename Derived, typename T>
inline size_t EventBase<Derived, T>::Nsys() const{
    return _state.nsys();
}

template<typename Derived, typename T>
inline size_t EventBase<Derived, T>::counter() const{
    return _counter;
}

template<typename Derived, typename T>
inline const std::vector<T>& EventBase<Derived, T>::args() const{
    return _args;
}

template<typename Derived, typename T>
Event<T>* EventBase<Derived, T>::clone() const{
    return new Derived(*THIS_C);
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_temporal() const{
    return THIS_C->is_temporal_impl();
}

template<typename Derived, typename T>
bool EventBase<Derived, T>::is_stop_event() const{
    return THIS_C->stop_impl();
}

template<typename Derived, typename T>
bool EventBase<Derived, T>::is_lethal() const{
    return THIS_C->kill_impl();
}

template<typename Derived, typename T>
inline int EventBase<Derived, T>::direction() const{
    return this->_direction;
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_located() const{
    return _is_located;
}

template<typename Derived, typename T>
inline const T& EventBase<Derived, T>::t_start() const{
    return _start;
}

template<typename Derived, typename T>
inline const EventState<T>* EventBase<Derived, T>::state() const{
    return &_state;
}

template<typename Derived, typename T>
void EventBase<Derived, T>::set_args(const T* args, size_t size){
    _args.resize(size);
    copy_array(_args.data(), args, size);
}

template<typename Derived, typename T>
inline void EventBase<Derived, T>::setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction){
    // checks that it has not been already setup
    // no other modifiers can be called if setup has not been called yet
    assert(!this->_is_setup && "Setup takes place only once");
    assert(abs(direction)==1 && "Invalid direction");
    set_args(args, nargs);
    _q_aux.resize(n_sys);
    _direction = direction;
    _state.resize(n_sys);
    _is_setup = true;
    _start = t_start;
}

template<typename Derived, typename T>
void EventBase<Derived, T>::deactivate(){
    _is_located = false;
    _state.triggered = false;
}

template<typename Derived, typename T>
bool EventBase<Derived, T>::locate(State<T> before, State<T> after, FuncLike<T> q, const void* obj){
    assert(this->_is_setup && "Call setup() method before trying to locate an event");
    assert((sgn(before.t(), after.t()) == this->_direction) && "Invalid direction");
    T t;
    _state.triggered = false;
    if (THIS_C->locate_impl(t, before, after, q, obj)){
        _state.set_t(t);
        _state.set_stepsize(after.habs());
        T* q_event = this->is_masked() ? _state.exposed_vector() : _state.true_vector();
        q(q_event, t, obj); //q_event has been set
        _is_located = true;
        return true;
    }else {
        _is_located = false;
        return false;
    }
}

template<typename Derived, typename T>
bool EventBase<Derived, T>::register_it(){
    if (_is_located){
        if (this->is_masked()){
            copy_array(_state.true_vector(), _state.exposed_vector(), _state.nsys());
            this->apply_mask(_state.true_vector(), _state.t(), _state.exposed_vector());
            _state.choose_true = !this->hides_mask();
        }else{
            _state.choose_true = true;
        }
        _state.triggered = true;
        THIS->register_impl();
        return true;
    }else{
        assert(!_state.triggered && "Report Bug"); //Sanity check. Should not need to set _state.triggered = false;
        return false;
    }
}

template<typename Derived, typename T>
void EventBase<Derived, T>::reset(){
    THIS->reset_impl();
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const{
    static_assert(false, "static override");
    return false;
}

template<typename Derived, typename T>
inline void EventBase<Derived, T>::register_impl(){
    _counter++;
}

template<typename Derived, typename T>
inline void EventBase<Derived, T>::reset_impl(){
    _counter = 0;
    _state.choose_true = true;
    _state.triggered = false;
}

template<typename Derived, typename T>
inline void EventBase<Derived, T>::mask_impl(T* out, const T& t, const T* q) const{
    assert(this->is_masked_impl() && "Default mask() implementation requires that mask != nullptr");
    this->_mask(out, t, q, _args.data(), _obj);
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_masked_impl() const{
    return _mask != nullptr;
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::is_temporal_impl() const{
    return false;
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::stop_impl() const{
    return false;
}

template<typename Derived, typename T>
inline bool EventBase<Derived, T>::kill_impl() const{
    return false;
}

// PreciseEvent implementations

template<typename T, typename Derived>
PreciseEvent<T, Derived>::PreciseEvent(std::string name, ObjFun<T> when, int dir, Func<T> mask, bool hide_mask, T event_tol, const void* obj) : Base(name, mask, hide_mask, obj), _when(when), _sign_dir(dir), _event_tol(event_tol) {}

template<typename T, typename Derived>
inline T PreciseEvent<T, Derived>::obj_fun(const T& t, const T* q) const{
    return _when(t, q, this->args().data(), this->_obj);
}

template<typename T, typename Derived>
inline int PreciseEvent<T, Derived>::sign_change_dir() const{
    return _sign_dir;
}

template<typename T, typename Derived>
inline bool PreciseEvent<T, Derived>::locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const{
    T val1 = this->obj_fun(before.t(), before.vector());
    T val2 = this->obj_fun(after.t(), after.vector());

    int t_dir = this->direction();
    int d = this->sign_change_dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && val1 != 0){
        T* vec = this->_q_aux.data();

        auto obj_fun = [&](T t) LAMBDA_INLINE{
            q(vec, t, obj); // interpolate the state vector at time t, and pass the value on vec
            return this->obj_fun(t, vec);
        };

        t = bisect<T, RootPolicy::Right>(obj_fun, before.t(), after.t(), this->_event_tol);
        return true;
    }
    return false;
}

// PeriodicEvent implementations

template<typename T, typename Derived>
PeriodicEvent<T, Derived>::PeriodicEvent(std::string name, T period, Func<T> mask, bool hide_mask, const void* obj) : Base(name, mask, hide_mask, obj), _period(period) {}

template<typename T, typename Derived>
inline const T& PeriodicEvent<T, Derived>::period() const{
    return _period;
}

template<typename T, typename Derived>
inline size_t PeriodicEvent<T, Derived>::np() const{
    return _n;
}

template<typename T, typename Derived>
inline T PeriodicEvent<T, Derived>::delta_t_abs() const{
    return _n*_period;
}

template<typename T, typename Derived>
inline T PeriodicEvent<T, Derived>::delta_t() const{
    return _n*_period*this->direction();
}

template<typename T, typename Derived>
inline bool PeriodicEvent<T, Derived>::is_temporal_impl() const{
    return true;
}

template<typename T, typename Derived>
inline T PeriodicEvent<T, Derived>::get_t(size_t n) const{
    return this->t_start() + this->direction()*n*this->period();
}

template<typename T, typename Derived>
inline bool PeriodicEvent<T, Derived>::locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const{
    _n_aux = _n;
    int d = this->direction();

    while (get_t(++_n_aux)*d <= before.t()*d){}

    if (get_t(_n_aux)*d <= after.t()*d){
        t = get_t(_n_aux);
        return true;
    }else {
        return false;
    }
}

template<typename T, typename Derived>
inline void PeriodicEvent<T, Derived>::register_impl(){
    Base::register_impl();
    _n = _n_aux;
}

template<typename T, typename Derived>
inline void PeriodicEvent<T, Derived>::reset_impl(){
    Base::reset_impl();
    _n = _n_aux = 0;
}

// TmaxEvent implementations

template<typename T, typename Derived>
inline void TmaxEvent<T, Derived>::set_goal(T tmax){
    _t_goal = tmax;
}

template<typename T, typename Derived>
inline bool TmaxEvent<T, Derived>::goal_is_set() const{
    return is_finite(_t_goal);
}

template<typename T, typename Derived>
inline bool TmaxEvent<T, Derived>::locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const{
    if (!goal_is_set()) {return false;}

    int d = this->direction();

    if ((before.t()*d < _t_goal*d) && (_t_goal*d <= after.t()*d )){
        t = _t_goal;
        return true;
    }else {
        return false;
    }
}

template<typename T, typename Derived>
inline void TmaxEvent<T, Derived>::register_impl(){
    Base::register_impl();
    _t_goal = inf<T>();
}

template<typename T, typename Derived>
inline void TmaxEvent<T, Derived>::reset_impl(){
    Base::reset_impl();
    _t_goal = inf<T>();
}

template<typename T, typename Derived>
inline bool TmaxEvent<T, Derived>::stop_impl() const{
    return true;
}

template<typename T, typename Derived>
inline bool TmaxEvent<T, Derived>::is_temporal_impl() const{
    return true;
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

// EventView implementations

template<typename T>
EventView<T>::EventView(const AnyEvent<T>* events, const size_t* detection, size_t size) : Base(detection, size), event_data(events) {}

template<typename T>
template<std::integral Int>
const Event<T>* EventView<T>::operator[](Int i) const{
    return event_data[Base::operator[](i)].ptr();
}

template<typename T>
size_t EventView<T>::size() const{
    return Base::size();
}

// EventCollection implementations

template<typename T>
EventCollection<T>::EventCollection(const std::vector<const Event<T>*>& events) : EventCollection(events.data(), events.size()) {}

template<typename T>
EventCollection<T>::EventCollection(const Event<T>*const* events, size_t size) : _events(size), _event_idx(size), _event_idx_start(size+1), _canon_idx(size) {
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
    return *_events[i]->state();
}

template<typename T>
inline size_t EventCollection<T>::size() const{
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
void EventCollection<T>::setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction){
    for (size_t i=0; i<this->size(); i++){
        _events[i]->setup(t_start, args, nargs, n_sys, direction);
    }
}

template<typename T>
void EventCollection<T>::set_args(const T* args, size_t size){
    for (size_t i=0; i<this->size(); i++){
        _events[i]->set_args(args, size);
    }
}

template<typename T>
void EventCollection<T>::detect_all_between(State<T> before, State<T> after, FuncLike<T> q, const void* obj){
    if (this->size() == 0){
        return;
    }

    /*
     * PHASE 1: Detection and Sorting
     *
     * Locate all events that occur between the 'before' and 'after' states.
     * Store their indices in _event_idx, sorted by priority using insertion sort.
     * Priority is determined by _is_prioritized(): earlier time wins, ties broken by event index.
     *
     * After this phase: _event_idx contains indices of all detected events, sorted by time.
     */
    const int dir = sgn(before.t(), after.t());
    _N_detect = 0;
    for (size_t i=0; i<this->size(); i++){
        Event<T>* event_obj = _events[i].ptr();
        if (event_obj->locate(before, after, q, obj)){
            long int j = (long int)(_N_detect)-1;
            while (j>=0 && _is_prioritized(i, _event_idx[j], dir)){
                _event_idx[j+1] = _event_idx[j];
                j--;
            }
            _event_idx[j+1] = i;
            _N_detect++;
        }
    }

    /*
     * PHASE 2: Resolve Same-Time Conflicts
     *
     * When multiple non-pure-temporal events occur at the same time, only one can determine
     * the state vector. The 'mark' tracks the first non-pure-temporal event at each time.
     * Since events are sorted by index (for same-time events), the user controls which event
     * "wins" by ordering them appropriately in the event list.
     *
     * For subsequent non-pure-temporal events at the same time as 'mark':
     *   - discard_event() decides which to discard based on masking rules:
     *     - If mark is masked (and not hidden): discard any non-pure-temporal event
     *     - If mark is not masked or hides its mask: discard any masked event
     *
     * Note: A masked event does NOT automatically win over a non-masked event at the same time.
     * The first non-pure-temporal event (by index) becomes the mark and determines the rules.
     *
     * Pure-temporal events (no state dependency) are always kept since they don't conflict.
     * When time changes, mark resets to allow a new "winner" at the new time.
     *
     * After this phase: At each time, at most one non-pure-temporal event remains (plus any pure-temporal).
     */
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
                _events[_event_idx[i]]->deactivate();
                for (size_t j=i; j<_N_detect-1; j++){
                    _event_idx[j] = _event_idx[j+1];
                }
                _N_detect--;
                continue;
            }
        }
        i++;
    }

    /*
     * PHASE 3: Find Canon Event and Group by Time
     *
     * The "canon" event is the first masked event encountered. It determines the true
     * state vector after all events at its time are processed (the mask transforms the state).
     *
     * This phase also builds time groups: _event_idx_start[k] marks where group k begins.
     * Events are grouped by identical detection times for iteration purposes.
     *
     * The loop breaks immediately after finding the first masked event (canon).
     *
     * After this phase: _canon_idx points to the canon event (or size() if none),
     * and time groups are partially built up to and including the canon's time.
     */
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

    /*
     * PHASE 4: Discard Events After Canon Time
     *
     * If a canon (masked) event was found, any events occurring AFTER the canon's time
     * must be discarded. The solver will restart from the canon's time with the transformed
     * state, so those later events will be re-detected in subsequent steps.
     *
     * First, skip any remaining events at the SAME time as canon (they're kept).
     * Then deactivate all events at LATER times.
     *
     * tmp/j tracks where the valid events end; _N_detect is updated to exclude deactivated ones.
     */
    size_t j=i;
    while (j<_N_detect && (state(_event_idx[j]).t()== state(_event_idx[i-1]).t())){
        j++;
    }
    size_t tmp = j;
    _event_idx_start[_Nt] = j;
    for (; j<_N_detect; j++){
        const_cast<Event<T>&>(event(_event_idx[j])).deactivate();
    }

    /*
     * PHASE 5: Register Events
     *
     * Call register_it() on all events that were successfully located (not deactivated).
     * This applies masks (copying exposed_vector to true_vector and transforming),
     * increments event counters, and performs any event-specific registration logic.
     *
     * Finally, update _N_detect to reflect only the kept events and reset the iterator.
     */
    for (i=0; i<size(); i++){
        if (_events[i]->is_located()) {
            _events[i]->register_it();
        }
    }
    _N_detect = tmp;
    _iter = 0;
}

template<typename T>
EventView<T> EventCollection<T>::event_view() const{
    return (_iter < _Nt) ? EventView<T>(_events.data(), _event_idx.data() + _event_idx_start[_iter], _event_idx_start[_iter+1] - _event_idx_start[_iter]) : EventView<T>(nullptr, nullptr, 0);
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
    if (const auto* event = this->canon_event()){
        return event->state();
    }else {
        return nullptr;
    }
}

template<typename T>
void EventCollection<T>::reset(){

    for (size_t i=0; i<this->size(); i++){
        _events[i]->reset();
        _event_idx[i] = 0;
        _event_idx_start[i] = 0;
    }

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
