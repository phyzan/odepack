#ifndef EVENTS_IMPL_HPP
#define EVENTS_IMPL_HPP

#include "Events.hpp"
#include "../Tools_impl.hpp"

// "../Tools_impl.hpp" is required as explicit instanciation
// of bisect<T> is needed.

namespace ode{

// EventBase implementations

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
EventBase<Derived, T, MaskFunc>::EventBase(std::string name, MaskFunc mask, bool hide_mask) : _name(std::move(name)), _mask(std::move(mask)), _hide_mask(hide_mask) {
    if (_name.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
const std::string& EventBase<Derived, T, MaskFunc>::name() const{
    return _name;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
constexpr bool EventBase<Derived, T, MaskFunc>::is_masked() const{
    if constexpr (std::is_same_v<MaskFunc, std::nullptr_t>){
        return false;
    } else if constexpr (std::is_pointer_v<MaskFunc>){
        return _mask != nullptr;
    } else {
        return true;
    }
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
bool EventBase<Derived, T, MaskFunc>::hides_mask() const{
    return _hide_mask && this->is_masked();
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::apply_mask(T* out, const T& t, const T* q) const{
    assert(this->is_masked() && "Default mask() implementation requires that mask != nullptr");
    if constexpr (std::is_same_v<MaskFunc, std::nullptr_t>){
        assert(false && "apply_mask called when MaskFunc is std::nullptr_t");
    } else {
        _mask(out, t, q, this->args().data());
    }
}


template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
size_t EventBase<Derived, T, MaskFunc>::Nsys() const{
    return worker.size();
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
size_t EventBase<Derived, T, MaskFunc>::counter() const{
    return _counter;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
const std::vector<T>& EventBase<Derived, T, MaskFunc>::args() const{
    return _args;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
Event<T>* EventBase<Derived, T, MaskFunc>::clone() const{
    return new Derived(*THIS);
}


template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
int EventBase<Derived, T, MaskFunc>::direction() const{
    return this->_direction;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
bool EventBase<Derived, T, MaskFunc>::is_located() const{
    return _is_located;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
const T& EventBase<Derived, T, MaskFunc>::t_start() const{
    return _start;
}


template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::set_args(const T* args, size_t size){
    _args.resize(size);
    ndspan::copy_array(_args.data(), args, size);
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction){
    // checks that it has not been already setup
    // no other modifiers can be called if setup has not been called yet
    assert(!this->_is_setup && "Setup takes place only once");
    assert(abs(direction)==1 && "Invalid direction");
    set_args(args, nargs);
    worker.resize(n_sys);
    _direction = direction;
    _is_setup = true;
    _start = t_start;
}


template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
template<StateInterp<T> Callable>
bool EventBase<Derived, T, MaskFunc>::locate_state(T& out, State<T> before, State<T> after, Callable&& obj_fun){
    assert(this->_is_setup && "Call setup() method before trying to locate an event");
    assert((sgn(before.t(), after.t()) == this->_direction) && "Invalid direction");
    if (THIS->locate_impl(out, before, after, obj_fun)){
        _is_located = true;
        return true;
    }else {
        _is_located = false;
        return false;
    }
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
bool EventBase<Derived, T, MaskFunc>::locate(T& out, State<T> before, State<T> after, const EventInterp<T>& interp){
    return this->locate_state(out, before, after, interp);
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
bool EventBase<Derived, T, MaskFunc>::lock(){
    if (_is_located){
        THIS->register_impl();
        return true;
    }else{
        return false;
    }
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::reset(int direction){
    THIS->reset_impl(direction);
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
template<StateInterp<T> Callable>
bool EventBase<Derived, T, MaskFunc>::locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const{
    static_assert(false, "static override");
    return false;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::register_impl(){
    _counter++;
}

template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc>
void EventBase<Derived, T, MaskFunc>::reset_impl(int direction){
    _counter = 0;
    _is_located = false;
    _direction = (direction == 0) ? _direction : direction;
}

// PreciseEvent implementations

template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc, typename Derived>
PreciseEvent<T, Target, MaskFunc, Derived>::PreciseEvent(std::string name, Target when, T event_tol, int dir, MaskFunc mask, bool hide_mask) : Base(std::move(name), std::move(mask), hide_mask), target(std::move(when)), crossing_dir(dir), ftol(event_tol) {}

template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc, typename Derived>
T PreciseEvent<T, Target, MaskFunc, Derived>::obj_fun(const T& t, const T* q) const{
    return target(t, q, this->args().data());
}

template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc, typename Derived>
int PreciseEvent<T, Target, MaskFunc, Derived>::sign_change_dir() const{
    return crossing_dir;
}

template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc, typename Derived>
template<StateInterp<T> Callable>
bool PreciseEvent<T, Target, MaskFunc, Derived>::locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const{
    T val1 = this->obj_fun(before.t(), before.vector());
    T val2 = this->obj_fun(after.t(), after.vector());

    int t_dir = this->direction();
    int d = this->sign_change_dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && (abs<T>(val1) > ftol)){
        T* vec = this->worker.data();

        auto obj_fun_scalar = [&](T t) LAMBDA_INLINE{
            obj_fun(vec, t); // interpolate the state vector at time t, and pass the value on vec
            return this->obj_fun(t, vec);
        };

        t = bisect<T, RootPolicy::Right>(obj_fun_scalar, before.t(), after.t(), this->ftol);
        return true;
    }
    return false;
}

// PeriodicEvent implementations

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
PeriodicEvent<T, MaskFunc, Derived>::PeriodicEvent(std::string name, T period, MaskFunc mask, bool hide_mask) : Base(name, std::move(mask), hide_mask), _period(period) {}

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
const T& PeriodicEvent<T, MaskFunc, Derived>::period() const{
    return _period;
}

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
T PeriodicEvent<T, MaskFunc, Derived>::get_t(size_t n) const{
    return this->t_start() + (this->direction()*n*this->period());
}

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
template<StateInterp<T> Callable>
bool PeriodicEvent<T, MaskFunc, Derived>::locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const{
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

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
void PeriodicEvent<T, MaskFunc, Derived>::register_impl(){
    Base::register_impl();
    _n = _n_aux;
}

template<typename T, OptionalRhsFunc<T> MaskFunc, typename Derived>
void PeriodicEvent<T, MaskFunc, Derived>::reset_impl(int direction){
    Base::reset_impl(direction);
    _n = _n_aux = 0;
}


template<typename T>
EventCollection<T>::EventCollection(const std::vector<const Event<T>*>& events) : EventCollection(events.data(), events.size()) {}

template<typename T>
EventCollection<T>::EventCollection(const Event<T>*const* events, size_t size) : events(size), event_times(size), detection_order(size), located(size) {

    if (size == 0){return;}

    for (size_t i=0; i<size; i++){
        if (idx_of_name.find(events[i]->name()) != idx_of_name.end()){
            throw std::runtime_error("Duplicate Event name not allowed: " + events[i]->name());
        }
        idx_of_name[events[i]->name()] = static_cast<int>(i);
        this->events[i].steal(events[i]->clone());
    }
}

template<typename T>
const Event<T>& EventCollection<T>::event(size_t event_idx) const{
    return *events[event_idx].get();
}


template<typename T>
size_t EventCollection<T>::event_idx(const std::string& name) const{
    auto it = idx_of_name.find(name);
    if (it != idx_of_name.end()){
        return it->second;
    }
    return -1;
}

template<typename T>
size_t EventCollection<T>::size() const{
    return events.size();
}

template<typename T>
size_t EventCollection<T>::detection_size() const{
    return detections;
}

template<typename T>
void EventCollection<T>::setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction){
    worker.resize(n_sys);
    masked_data.masked_vector.resize(n_sys);
    for (size_t i=0; i<this->size(); i++){
        events[i]->setup(t_start, args, nargs, n_sys, direction);
    }
}

template<typename T>
void EventCollection<T>::set_args(const T* args, size_t size){
    for (size_t i=0; i<this->size(); i++){
        events[i]->set_args(args, size);
    }
}

template<typename T>
template<StateInterp<T> Callable>
bool EventCollection<T>::detect_all_between(State<T> before, State<T> after, Callable&& obj_fun) {
    if (this->size() == 0){
        return false;
    }

    detections = 0;
    has_masked_state = false;
    located.fill(false);

    // Locate all events and record which ones were found
    for (size_t i=0; i<this->size(); i++){
        if (events[i]->locate(event_times[i], before, after, getEventInterp<T>(obj_fun))){
            located[i] = true;
            detection_order[detections++] = i;
        }
    }

    if (detections == 0){
        return false;
    }

    // Sort detected events by time (respecting integration direction), with index as tiebreaker
    int dir = events[0]->direction();
    std::sort(detection_order.data(), detection_order.data() + detections,
        [&](size_t a, size_t b){
            T ta = dir * event_times[a];
            T tb = dir * event_times[b];
            return (ta < tb) || (ta == tb && a < b);
        });

    // Remove simultaneous events, keeping only the first one (lowest index)
    size_t write = 1;
    for (size_t read = 1; read < detections; read++){
        if (event_times[detection_order[read]] != event_times[detection_order[read - 1]]){
            detection_order[write++] = detection_order[read];
        } else {
            // Discard this event - mark as not located
            located[detection_order[read]] = false;
        }
    }

    detections = write;

    // Find the first masked event and truncate detections after it
    size_t first_masked = detections;
    for (size_t i=0; i<detections; i++){
        size_t idx = detection_order[i];
        if (events[idx]->is_masked()){
            first_masked = i;
            break;
        }
    }

    // If a masked event was found, compute the masked state and truncate
    if (first_masked < detections){
        size_t masked_idx = detection_order[first_masked];
        T t_mask = event_times[masked_idx];

        // Interpolate state at mask time
        obj_fun(worker.data(), t_mask);

        // Apply the mask transformation
        events[masked_idx]->apply_mask(masked_data.masked_vector.data(), t_mask, worker.data());
        masked_data.time = t_mask;
        masked_data.idx = masked_idx;
        has_masked_state = true;

        // Mark events after the masked event as not located
        for (size_t i = first_masked + 1; i < detections; i++){
            located[detection_order[i]] = false;
        }

        // Keep only events up to and including the first masked event
        detections = first_masked + 1;
    }

    return true;
}

template<typename T>
void EventCollection<T>::reset(int direction){

    for (size_t i=0; i<this->size(); i++){
        events[i]->reset(direction);
        located[i] = false;
    }

    detections = 0;
    has_masked_state = false;
}


} // namespace ode

#endif // EVENTS_IMPL_HPP