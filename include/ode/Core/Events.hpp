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
 * - EventBase<Derived, T>: CRTP base with common implementation
 *   - PreciseEvent<T>: Zero-crossing detection with bisection
 *   - PeriodicEvent<T>: Fixed-interval temporal events
 * - ObjectOwningEvent<EventType, ObjType>: Wrapper owning user data
 * - EventCollection<T>: Container managing multiple events
 * - EventView<T>: View into detected events at a time point
 */

#include "../Tools.hpp"

namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================

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

    /// @brief Check if event is temporal (time-based) and has no mask.
    virtual bool                    is_pure_temporal() const = 0;

    /// @brief Get the number of times this event has been triggered.
    virtual size_t                  counter() const = 0;

    /// @brief Get the additional arguments for event functions.
    virtual const std::vector<T>&   args() const = 0;

    /// @brief Create a dynamically allocated copy of this event.
    virtual Event<T>*               clone() const = 0;

    /// @brief Check if this is a temporal (time-based) event.
    virtual bool                    is_temporal() const = 0;

    /// @brief Get the integration direction (+1 forward, -1 backward).
    virtual int                     direction() const = 0;

    /// @brief Check if the event was located in the last detection pass.
    virtual bool                    is_located() const = 0;

    /// @brief Get the event state containing trigger time and state vector.
    virtual const EventState<T>*    state() const = 0;

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

    /// @brief Mark this event as not located (clear detection state).
    virtual void deactivate() = 0;

    /**
     * @brief Attempt to locate when this event triggers between two states.
     * @param before State at the start of the interval.
     * @param after  State at the end of the interval.
     * @param q      Interpolation function for intermediate states.
     * @param obj    User object pointer passed to callbacks.
     * @return True if event was detected in the interval.
     */
    virtual bool locate(State<T> before, State<T> after, FuncLike<T> q, const void* obj) = 0;

    /// @brief Register the event (increment counter, apply mask, etc.).
    virtual bool register_it() = 0;

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
template<typename Derived, typename T>
class EventBase : public Event<T>{

public:

    // ACCESSORS

    /// @brief Get the event name.
     const std::string&       name() const;
    /// @brief Check if this event has a mask function.
     bool                     is_masked() const;
    /// @brief Check if the masked state is hidden (showing original instead).
     bool                     hides_mask() const;
    /// @brief Apply the mask transformation to a state.
     void                     apply_mask(T* out, const T& t, const T* q) const;
    /// @brief Check if event is temporal with no mask.
     bool                     is_pure_temporal() const;
    /// @brief Get the ODE system size.
     size_t                   Nsys() const;
    /// @brief Get the trigger count.
     size_t                   counter() const;
    /// @brief Get the event arguments.
     const std::vector<T>&    args() const;
    /// @brief Clone this event.
    Event<T>*                       clone() const;
    /// @brief Check if this is a temporal event.
     bool                     is_temporal() const;
    /// @brief Get the integration direction.
     int                      direction() const;
    /// @brief Check if event was located in last detection pass.
     bool                     is_located() const;
    /// @brief Get the integration start time.
     const T&                 t_start() const;
    /// @brief Get the event state.
     const EventState<T>*     state() const;

    // MODIFIERS

    /// @brief Update event arguments.
    void                            set_args(const T* args, size_t size);
    /// @brief Initialize event for use with solver.
     void                     setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);
    /// @brief Clear the located state.
    void                            deactivate();
    /// @brief Attempt to locate event in an interval.
    bool                            locate(State<T> before, State<T> after, FuncLike<T> q, const void* obj);
    /// @brief Register the event after location.
    bool                            register_it();
    /// @brief Reset to initial state.
    void                            reset(int direction = 0);

protected:

    using Main = EventBase<Derived, T>;

    /**
     * @brief Construct an event with optional mask.
     * @param name      Unique event name.
     * @param mask      Optional mask function to transform state at trigger.
     * @param hide_mask If true, hide the masked state and show original.
     * @param obj       Optional user object pointer for callbacks.
     */
    EventBase(std::string name, Func<T> mask, bool hide_mask, const void* obj = nullptr);

    DEFAULT_RULE_OF_FOUR(EventBase);

    // ================ STATIC OVERRIDE (REQUIRED) ======================

    /**
     * @brief Locate when event triggers in an interval (must be overridden).
     * @param[out] t      Output time of event trigger.
     * @param[in]  before State at interval start.
     * @param[in]  after  State at interval end.
     * @param[in]  q      Interpolation function.
     * @param[in]  obj    User object pointer.
     * @return True if event was found in interval.
     */
     bool locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    // ============= STATIC OVERRIDE (OPTIONAL) ================

    /// @brief Called when event is registered. Override to add custom logic.
     void register_impl();
    /// @brief Called on reset. Override to reset derived state.
     void reset_impl(int direction);
    /// @brief Apply mask to state. Override for custom mask behavior.
     void mask_impl(T* out, const T& t, const T* q) const;
    /// @brief Check if masked. Override to customize mask detection.
     bool is_masked_impl() const;
    /// @brief Check if temporal. Override to mark as temporal event.
     bool is_temporal_impl() const;
    //==========================================================

private:
    std::string             _name;
    std::vector<T>          _args = {};
    EventState<T>           _state;
    T                       _start = inf<T>();
protected:
    /// @brief Auxiliary array for intermediate calculations.
    mutable Array1D<T>      _q_aux;
    /// @brief User object pointer passed to callbacks.
    const void*             _obj = nullptr;
private:
    size_t                  _counter = 0;
    Func<T>                 _mask = nullptr;
    int                     _direction;
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
template<typename T, typename Derived = void>
class PreciseEvent : public EventBase<GetDerived<PreciseEvent<T, Derived>, Derived>, T>{

    using Base = EventBase<GetDerived<PreciseEvent<T, Derived>, Derived>, T>;
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
    PreciseEvent(std::string name, ObjFun<T> when, int dir=0, Func<T> mask=nullptr, bool hide_mask=false, T event_tol=1e-12, const void* obj = nullptr);

    /// @brief Evaluate the objective function at given time and state.
     T    obj_fun(const T& t, const T* q) const;

    /// @brief Get the required sign change direction (+1, -1, or 0).
     int  sign_change_dir() const;

protected:

    /// @brief Locate zero crossing using bisection.
     bool locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    ObjFun<T>           _when = nullptr;      ///< Objective function to monitor.
    int                 _sign_dir = 1;        ///< Required crossing direction.
    T                   _event_tol;           ///< Bisection tolerance.

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
template<typename T, typename Derived = void>
class PeriodicEvent : public EventBase<GetDerived<PeriodicEvent<T, Derived>, Derived>, T>{

    using Base = EventBase<GetDerived<PeriodicEvent<T, Derived>, Derived>, T>;
    friend Base;

public:

    /**
     * @brief Construct a periodic event.
     * @param name      Unique event name.
     * @param period    Time interval between triggers.
     * @param mask      Optional mask function to transform state at trigger.
     * @param hide_mask If true, hide the masked state and show original.
     * @param obj       Optional user object pointer for callbacks.
     */
    PeriodicEvent(std::string name, T period, Func<T> mask=nullptr, bool hide_mask=false, const void* obj = nullptr);

    /// @brief Get the period (time between triggers).
     const T& period() const;

    /// @brief Get the number of periods elapsed since start.
     size_t   np() const;

    /// @brief Get the absolute time elapsed (n * period).
     T        delta_t_abs() const;

    /// @brief Get the signed time elapsed (n * period * direction).
     T        delta_t() const;

protected:

    /// @brief Returns true (this is a temporal event).
     bool     is_temporal_impl() const;

    /// @brief Compute time of n-th trigger.
     T        get_t(size_t n) const;

    /// @brief Locate next periodic trigger in interval.
     bool     locate_impl(T& t, State<T> before, State<T> after, FuncLike<T> q, const void* obj) const;

    /// @brief Update period counter on registration.
     void     register_impl();

    /// @brief Reset period counter.
     void     reset_impl(int direction);

private:

    T _period;                  ///< Time interval between triggers.
    size_t _n = 0;              ///< Current period count.
    mutable size_t _n_aux = 0;  ///< Auxiliary counter for detection.

};


/**
 * @brief Wrapper that owns the user object passed to event callbacks.
 *
 * Normally events store a pointer to user data, requiring the user to
 * manage lifetime. This wrapper owns a copy of the user object, ensuring
 * it stays valid as long as the event exists.
 *
 * @tparam EventType The base event type to wrap.
 * @tparam ObjType   The user object type to own.
 */
template<typename EventType, typename ObjType>
class ObjectOwningEvent : public EventType{

public:

    /**
     * @brief Construct owning event with user object.
     * @param obj  User object to copy and own.
     * @param args Arguments forwarded to EventType constructor.
     */
    template<typename... Args>
    ObjectOwningEvent(const ObjType& obj, Args&&... args);
    ObjectOwningEvent(const ObjectOwningEvent& other);
    ObjectOwningEvent(ObjectOwningEvent&& other) noexcept;
    ObjectOwningEvent& operator=(const ObjectOwningEvent& other);
    ObjectOwningEvent& operator=(ObjectOwningEvent&& other) noexcept;
    ObjectOwningEvent<EventType, ObjType>* clone() const override;

protected:

    ObjType _object;  ///< Owned copy of user object.

};

/**
 * @brief Determine if an event should be discarded based on masking rules.
 *
 * When multiple events occur at the same time, this determines which to keep.
 * The 'mark' event is the first non-pure-temporal event at that time.
 *
 * @param event The event to potentially discard.
 * @param mark  The reference event determining the rules.
 * @return True if event should be discarded.
 */
template<typename T>
bool discard_event(const Event<T>& event, const Event<T>& mark);

/// @brief Type-erased event container.
template<typename T>
using AnyEvent = PolyWrapper<Event<T>>;

/**
 * @brief View into a subset of detected events at a single time point.
 *
 * Provides indexed access to events that triggered at the same time
 * during a detection pass.
 *
 * @tparam T Scalar type for computations.
 */
template<typename T>
class EventView : private View1D<size_t>{

    using Base = View1D<size_t>;

public:

    /// @brief Construct a view over detected events.
    EventView(const AnyEvent<T>* events, const size_t* detection, size_t size);

    /// @brief Access event by index within this view.
    const Event<T>* operator[](size_t i) const;

    /// @brief Get number of events in this view.
    size_t size() const;

    const AnyEvent<T>* event_data;  ///< Pointer to event storage.

};


/**
 * @brief Container managing multiple events during ODE integration.
 *
 * Handles event detection, sorting by time, conflict resolution for
 * simultaneous events, and iteration over detection results. Events
 * are cloned on construction so the collection owns its events.
 *
 * @tparam T Scalar type for computations.
 */
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

    /// @brief Get event by index.
     const Event<T>&      event(size_t i) const;

    /// @brief Get event state by index.
     const EventState<T>& state(size_t i) const;

    /// @brief Get total number of events.
     size_t               size() const;

    /// @brief Get number of events detected in last pass.
     size_t               detection_size() const;

    /// @brief Get number of distinct trigger times in last pass.
     size_t               detection_times() const;

    /// @brief Get the pointer to the first detected event in the next time group
    const Event<T>*             get_next_event() const;

    /// @brief Initialize all events for use with solver.
    void                        setup(T t_start, const T* args, size_t nargs, size_t n_sys, int direction);

    /// @brief Update arguments for all events.
    void                        set_args(const T* args, size_t size);

    /**
     * @brief Detect all events between two states.
     *
     * Locates events, sorts by time, resolves conflicts, and prepares
     * results for iteration.
     */
    void                        detect_all_between(State<T> before, State<T> after, FuncLike<T> q, const void* obj);
    /// @brief Get view of events at current iteration time.
    EventView<T>                event_view() const;
    /// @brief Get iterator to start of current time group indices.
    const size_t*               begin() const;
    /// @brief Get iterator to end of current time group indices.
    const size_t*               end() const;
    /// @brief Advance to next time group.
     bool                 next_result();
    /// @brief Reset iteration to first time group.
     void                 restart_iter();
    /// @brief Get the first masked event that determines the state transformation, if any.
    const Event<T>*             canon_event() const;
    /// @brief Get the state of the first masked event, if any.
    const EventState<T>*        canon_state() const;
    /// @brief Reset all events to initial state.
    void                        reset(int direction = 0);

private:

    bool _is_prioritized(size_t i, size_t j, int dir);

    Array1D<AnyEvent<T>>    _events;
    Array1D<size_t>         _event_idx;
    Array1D<size_t>         _event_idx_start;

    size_t                  _canon_idx;
    size_t                  _N_detect=0;
    size_t                  _Nt=0;
    size_t                  _iter=0;

};


} // namespace ode

#endif
