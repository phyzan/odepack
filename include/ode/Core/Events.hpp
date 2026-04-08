#ifndef EVENTS_HPP
#define EVENTS_HPP

/**
 * @file events.hpp
 * @brief Event detection system for ODE integration.
 *
 * This file defines the event detection infrastructure used during ODE solving.
 * Events can trigger at specific times, periodic intervals, or when a user-defined
 * function crosses zero. Events also support masking (state transformation at trigger).
 *
 * Class hierarchy:
 * - Event<T>: Abstract interface for all events
 * - EventBase<Derived, ...>: CRTP base with common implementation
 *   - PreciseEvent<T>: Zero-crossing detection with bisection
 *   - PeriodicEvent<T>: Fixed-interval temporal events
 * - ObjectOwningEvent<EventType, ObjType>: Wrapper owning user data
 * - EventCollection<T>: Container managing multiple events
 * - EventView<T>: View into detected events at a time point
 */

#include "../Tools.hpp"

namespace ode {

    using pbox::owner;

    using ndspan::copy_array;

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
class EventInterp{

public:

    virtual ~EventInterp() = default;

    virtual void operator()(T* out, const T& t) const = 0;
};

template<typename T, typename Callable>
class UnaryInterp : public EventInterp<T>{

public:

    UnaryInterp(Callable func) : _func(std::move(func)) {}

    void operator()(T* out, const T& t) const override{
        _func(out, t);
    }

private:

    Callable _func;

};


template<typename T, typename Callable>
inline UnaryInterp<T, std::decay_t<Callable>> getEventInterp(Callable&& func){
    return UnaryInterp<T, std::decay_t<Callable>>(std::forward<Callable>(func));
}


/**
 * @brief Abstract interface for ODE integration events.
 *
 * Events are triggered during integration when certain conditions are met.
 * They can modify the state (masking), stop the solver, or simply record
 * that a condition occurred.
 *
 * @tparam T Scalar type for computations.
 */
template<typename T>
class Event{

public:

    virtual ~Event() = default;

    // ACCESSORS

    /// @brief Get the unique name of this event.
    virtual const std::string&      name() const = 0;

    /// @brief Check if this event has a mask function that transforms the state.
    virtual bool                    is_masked() const = 0;

    /// @brief Check if the masked (transformed) state is hidden, showing only the original.
    virtual bool                    hides_mask() const = 0;

    /**
     * @brief Apply the mask transformation to a state vector.
     * @param[out] out Output array for transformed state.
     * @param[in]  t   Time at which to apply mask.
     * @param[in]  q   Input state vector to transform.
     */
    virtual void                    apply_mask(T* out, const T& t, const T* q) const = 0;

    /// @brief Get the number of times this event has been triggered.
    virtual size_t                  counter() const = 0;

    /// @brief Get the additional arguments for event functions.
    virtual const std::vector<T>&   args() const = 0;

    /// @brief Create a dynamically allocated copy of this event.
    virtual Event<T>*               clone() const = 0;

    /// @brief Get the integration direction (+1 forward, -1 backward).
    virtual int                     direction() const = 0;
    // MODIFIERS

    /**
     * @brief Initialize the event for use with a solver.
     * @param t_start  Starting time of integration.
     * @param args     Additional arguments for event functions.
     * @param nargs    Number of arguments.
     * @param n_sys    Size of the ODE system.
     * @param direction Integration direction (+1 or -1).
     */
    virtual void setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction) = 0;

    /**
     * @brief Update the event's argument array.
     * @param args Pointer to new argument values.
     * @param size Number of arguments.
     */
    virtual void set_args(const T* args, size_t size) = 0;

    /**
     * @brief Attempt to locate when this event triggers between two states.
     * @param before State at the start of the interval.
     * @param after  State at the end of the interval.
     * @param interp Interpolation object for intermediate states.
     * @return True if event was detected in the interval.
     */
    virtual bool locate(T& out, State<T> before, State<T> after, const EventInterp<T>& interp) = 0;

    /// @brief If the last locate() call returned true.
    virtual bool is_located() const = 0;

    /// @brief Register the event after it has been located. If the last locate() returned True, this finalizes
    /// the event detection and returns true. Otherwise returns false and nothing happens.
    virtual bool lock() = 0;

    /// @brief Reset the event to its initial state.
    virtual void reset(int direction = 0) = 0;
};


/**
 * @brief CRTP base class implementing the Event interface.
 *
 * Provides common functionality for all event types including state management,
 * masking, and registration. Derived classes implement locate_impl() to define
 * when the event triggers.
 *
 * @tparam Derived The derived event class (CRTP pattern).
 * @tparam T       Scalar type for computations.
 */
template<typename Derived, typename T, OptionalRhsFunc<T> MaskFunc = std::nullptr_t>
class EventBase : public Event<T>{

public:

    // ------------------------------ ACCESSORS -----------------------------------

    /// @brief Get the event name.
    const std::string&      name() const;

    /// @brief Check if this event has a mask function.
    constexpr bool          is_masked() const;

    /// @brief Check if the masked state is hidden (showing original instead).
    bool                    hides_mask() const;

    /// @brief Apply the mask transformation to a state.
    void                    apply_mask(T* out, const T& t, const T* q) const;

    /// @brief Get the ODE system size.
    size_t                  Nsys() const;

    /// @brief Get the trigger count.
    size_t                  counter() const;

    /// @brief Get the event arguments.
    const std::vector<T>&   args() const;

    /// @brief Clone this event.
    Event<T>*               clone() const;

    /// @brief Get the integration direction.
    int                     direction() const;

    /// @brief Check if event was located in last detection pass.
    bool                    is_located() const;

    /// @brief Get the integration start time.
    const T&                t_start() const;

    // ------------------------------ MODIFIERS -----------------------------------

    /// @brief Update event arguments.
    void                    set_args(const T* args, size_t size);

    /// @brief Initialize event for use with solver.
    void                    setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);

    /// @brief Attempt to locate event in an interval. obj_fun(T* out, T t) -> void fills the state vector at time t.
    template<StateInterp<T> Callable>
    bool                    locate_state(T& out, State<T> before, State<T> after, Callable&& obj_fun);

    /// @brief Attempt to locate event in an interval.
    bool                    locate(T& out, State<T> before, State<T> after, const EventInterp<T>& interp);

    /// @brief Register the event after location.
    bool                    lock();

    /// @brief Reset to initial state.
    void                    reset(int direction = 0);

protected:

    using Main = EventBase<Derived, T, MaskFunc>;

    /**
     * @brief Construct an event with optional mask.
     * @param name      Unique event name.
     * @param mask      Optional mask function to transform state at trigger.
     * @param hide_mask If true, hide the masked state and show original.
     * @param obj       Optional user object pointer for callbacks.
     */
    EventBase(std::string name, MaskFunc mask, bool hide_mask);

    DEFAULT_RULE_OF_FOUR(EventBase);

    // ================ STATIC OVERRIDE (REQUIRED) ======================

    /**
     * @brief Locate when event triggers in an interval (must be overridden).
     * @param[out] t      Output time of event trigger.
     * @param[in]  before State at interval start.
     * @param[in]  after  State at interval end.
     * @param[in]  obj_fun Interpolation function (T* out, const T& t) -> void to compute state at intermediate times (for masked events).
     * @return True if event was found in interval.
     */
    template<StateInterp<T> Callable>
    bool locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const;

    // ============= STATIC OVERRIDE (OPTIONAL) ================

    /// @brief Called when event is registered. Override to add custom logic.
    void register_impl();

    /// @brief Called on reset. Override to reset derived state.
    void reset_impl(int direction);

    /// @brief Check if temporal. Override to mark as temporal event.
    bool is_temporal_impl() const;
    //==========================================================

private:
    std::string             _name;
    std::vector<T>          _args;
    T                       _start;
protected:
    /// @brief Auxiliary array for intermediate calculations.
    mutable Array1D<T>      worker;
    /// @brief User object pointer passed to callbacks.
private:
    MaskFunc                _mask;
    size_t                  _counter = 0;
    int                     _direction = 0; //+1 forward, -1 backward
    bool                    _hide_mask = false;
    bool                    _is_setup = false;
    bool                    _is_located = false;

};


/**
 * @brief Event triggered when a user function crosses zero.
 *
 * Uses bisection to precisely locate the time when the objective function
 * crosses zero. The crossing direction can be specified to trigger only
 * on increasing, decreasing, or any crossing.
 *
 * @tparam T       Scalar type for computations.
 * @tparam Derived Optional derived class for further CRTP extension.
 */
template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc = std::nullptr_t, typename Derived = void>
class PreciseEvent : public EventBase<GetDerived<PreciseEvent<T, Target, MaskFunc, Derived>, Derived>, T, MaskFunc>{

    using Base = EventBase<GetDerived<PreciseEvent<T, Target, MaskFunc, Derived>, Derived>, T, MaskFunc>;
    friend Base;

public:

    /**
     * @brief Construct a zero-crossing event.
     * @param name      Unique event name.
     * @param when      Objective function: triggers when this crosses zero.
     * @param dir       Crossing direction: +1 (increasing), -1 (decreasing), 0 (any).
     * @param mask      Optional mask function to transform state at trigger.
     * @param hide_mask If true, hide the masked state and show original.
     * @param event_tol Tolerance for bisection root finding.
     * @param obj       Optional user object pointer for callbacks.
     */
    PreciseEvent(std::string name, Target when, T event_tol=1e-20, int dir=0, MaskFunc mask=nullptr, bool hide_mask=false);

    /// @brief Evaluate the objective function at given time and state.
    T    obj_fun(const T& t, const T* q) const;

    /// @brief Get the required sign change direction (+1, -1, or 0).
    int  sign_change_dir() const;

protected:

    /// @brief Locate zero crossing using bisection.
    template<StateInterp<T> Callable>
    bool locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const;

    Target  target = nullptr;      ///< Objective function to monitor.
    int     crossing_dir = 1;        ///< Required crossing direction.
    T       ftol;           ///< Bisection tolerance.

};


/**
 * @brief Event triggered at regular time intervals.
 *
 * Fires at times t_start + n*period for n = 1, 2, 3, ...
 * This is a temporal event (state-independent).
 *
 * @tparam T       Scalar type for computations.
 * @tparam Derived Optional derived class for further CRTP extension.
 */
template<typename T, OptionalRhsFunc<T> MaskFunc = std::nullptr_t, typename Derived = void>
class PeriodicEvent : public EventBase<GetDerived<PeriodicEvent<T, MaskFunc, Derived>, Derived>, T, MaskFunc>{

    using Base = EventBase<GetDerived<PeriodicEvent<T, MaskFunc, Derived>, Derived>, T, MaskFunc>;
    friend Base;

public:

    /**
     * @brief Construct a periodic event.
     * @param name      Unique event name.
     * @param period    Time interval between triggers.
     * @param mask      Optional mask function to transform state at trigger.
     * @param hide_mask If true, hide the masked state and show original.
     */ 
    PeriodicEvent(std::string name, T period, MaskFunc mask=nullptr, bool hide_mask=false);

    /// @brief Get the period (time between triggers).
    const T&    period() const;

protected:

    /// @brief Returns true (this is a temporal event).
    bool        is_temporal_impl() const;

    /// @brief Compute time of n-th trigger.
    T           get_t(size_t n) const;

    /// @brief Locate next periodic trigger in interval.
    template<StateInterp<T> Callable>
    bool        locate_impl(T& t, State<T> before, State<T> after, Callable&& obj_fun) const;

    /// @brief Update period counter on registration.
    void        register_impl();

    /// @brief Reset period counter.
    void        reset_impl(int direction);

private:

    T _period;                  ///< Time interval between triggers.
    size_t _n = 0;              ///< Current period count.
    mutable size_t _n_aux = 0;  ///< Auxiliary counter for detection.

};


template<typename T>
struct MaskedState{
    Array1D<T> masked_vector;
    T time;
    size_t idx; //index of event that produced this masked state
};


template<typename T>
class EventCollection{

public:

    /// @brief Construct from array of event pointers.
    EventCollection(const Event<T>*const* events, size_t size);
    /// @brief Construct from vector of event pointers.
    EventCollection(const std::vector<const Event<T>*>& events);
    EventCollection() = default;
    DEFAULT_RULE_OF_FOUR(EventCollection)
    ~EventCollection() = default;

    // ------------------------ ACCESSORS ----------------------------

    /// @brief Get event by index.
    const Event<T>&         event(size_t event_idx) const;

    T                       get_time(size_t detection_idx) const{
        assert(detection_idx < detections && "Out of bounds detection_idx requested in get_time");
        return event_times[detection_order[detection_idx]];
    }

    const Event<T>*         get_event(size_t detection_idx) const{
        if (detection_idx >= detections){
            return nullptr;
        }else{
            return events[detection_order[detection_idx]].get();
        }
    }

    bool                    is_located(size_t event_idx) const{
        return located[event_idx];
    }

    /// @brief Get the index of the event with the given name, or -1 if not found.
    size_t                  event_idx(const std::string& name) const;

    size_t                  get_event_idx(size_t detection_idx) const{
        assert(detection_idx < detections && "Detection index out of bounds in get_event_idx");
        return detection_order[detection_idx];
    }

    /// @brief Get total number of events.
    size_t                  size() const;

    /// @brief Get number of events detected in last pass.
    size_t                  detection_size() const;

    /// @brief Initialize all events for use with solver.
    void                    setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);

    /// @brief Update arguments for all events.
    void                    set_args(const T* args, size_t size);

    /**
     * @brief Detect all events between two states.
     *
     * Locates events, sorts by time, resolves conflicts, and prepares
     * results for iteration.
        * @param before State at interval start.
        * @param after  State at interval end.
        * @param obj_fun Interpolation function (T* out, const T& t) -> void to compute state at intermediate times (for masked events).
     */
    template<StateInterp<T> Callable>
    bool                    detect_all_between(State<T> before, State<T> after, Callable&& obj_fun);

    const MaskedState<T>*   masked_state() const{
        return has_masked_state ? &masked_data : nullptr;
    }

    /// @brief Reset all events to initial state.
    void                    reset(int direction = 0);

private:
    std::unordered_map<std::string, size_t> idx_of_name;    // Map from event names to indices for quick lookup.
    Array1D<owner<Event<T>>>                events;
    Array1D<T>                              event_times;    // Detection time of each event. If located[i] is false, this value is invalid.
    Array1D<T>                              worker;       // Auxiliary array for intermediate calculations during detection.
    Array1D<size_t>                         detection_order;  // Indices of events sorted by detection time after detect_all_between() is called.
    Array1D<bool>                           located;
    MaskedState<T>                          masked_data;
    size_t                                  detections = 0;
    bool                                    has_masked_state = false;
};


template<typename T>
struct EventState{
    const Event<T>* event = nullptr; // Pointer to the event that triggered, or nullptr if no event is active
    size_t idx = 0; //index of the event in the event collection. garbage if the event is not active
    bool is_masked = false;
    bool active = false; //true if the solver is currently at this event

    operator bool() const{
        return active;
    }
};

} // namespace ode

#endif
