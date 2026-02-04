#ifndef RICH_SOLVER_HPP
#define RICH_SOLVER_HPP

/**
 * @file RichBase.hpp
 * @brief Extended ODE solver class with event detection and dense output interpolation.
 *
 * This file defines the RichSolver template class, which extends BaseSolver with
 * additional functionality for event detection and continuous interpolation.
 * RichSolver maintains a linked chain of interpolators that enables dense output
 * across integration steps, and supports event-driven integration where the solver
 * can detect and respond to user-defined events during integration.
 *
 * Key features beyond BaseSolver:
 * - Event detection with configurable triggering conditions
 * - Continuous dense output interpolation across multiple steps
 * - Automatic handling of state discontinuities at events
 */

#include "SolverBase.hpp"

#define EVENTS const std::vector<const Event<T>*>&

namespace ode{

/**
 * @brief Internal interpolation function used for event detection callbacks.
 * @param[out] res Output array for interpolated state (size Nsys).
 * @param[in]  t   Time to interpolate at.
 * @param[in]  obj Pointer to the RichSolver instance.
 */
template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void interp_func(T* res, const T& t, const void* obj);

/**
 * @brief Extended ODE solver with event detection and dense output capabilities.
 *
 * RichSolver extends BaseSolver to provide event-driven integration and continuous
 * dense output interpolation. It maintains an internal collection of events that
 * are monitored during integration, and builds a chain of interpolators that
 * enables querying the solution at any time within the integration history.
 *
 * The solver automatically handles state discontinuities that may occur at events,
 * and provides facilities for advancing to the next event or querying which events
 * are currently active.
 *
 * @tparam Derived The derived solver class (CRTP pattern).
 * @tparam T       Scalar type for computations (e.g., double, float).
 * @tparam N       System size at compile time. Use 0 for runtime-sized systems.
 * @tparam SP      Solver policy (must be RichStatic or RichVirtual for this class).
 * @tparam RhsType Type of the right-hand side function.
 * @tparam JacType Type of the Jacobian function (or nullptr_t if not provided).
 *
 * @note This class uses static polymorphism (CRTP).
 *       The solver policy must be RichStatic or RichVirtual for this class to be used.
 */
template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
class RichSolver : public BaseSolver<Derived, T, N, SP, RhsType, JacType>{

    using Base = BaseSolver<Derived, T, N, SP, RhsType, JacType>;

    friend Base; // So that Base can access specific private methods for static override
    friend void interp_func<Derived, T, N, SP, RhsType, JacType>(T*, const T&, const void*);

public:

    // ACCESSORS

    /**
     * @brief Get the current state vector that integration continues from.
     *
     * When at an event that applies a mask, this returns the new masked state
     * (the state that integration will continue from), regardless of whether the event was declared
     * with a hidden mask or not. Otherwise, returns the base solver state, equivalent to Base::vector().
     *
     * @return View of the true state vector (size Nsys).
     */
    inline View1D<T, N>                     true_vector() const;

    /**
     * @brief Get a view of the currently active events.
     *
     * Returns information about events that have triggered at the current time.
     * If not at an event, returns an empty view.
     *
     * @return EventView containing active events, or empty if none.
     */
    EventView<T>                            current_events() const;

    /**
     * @brief Get a reference to the event collection.
     * @return Const reference to the internal EventCollection.
     */
    const EventCollection<T>&               event_col() const;

    /**
     * @brief Get the dense output interpolator.
     *
     * Returns a pointer to the linked interpolator chain that provides dense
     * output. Only valid if interpolation has been started via start_interpolation().
     *
     * @return Pointer to the interpolator, or nullptr if not interpolating.
     */
    const Interpolator<T, N>*               interpolator() const;

    /**
     * @brief Check if dense output interpolation is active.
     * @return True if interpolation data is being collected.
     */
    bool                                    is_interpolating() const;

    /**
     * @brief Get a State object representing the current solver position.
     *
     * If at an event with a visible mask, returns the masked state.
     * If at an event with a hidden mask, returns the pre-mask state.
     * Otherwise returns the base solver state.
     *
     * @return State object with current time and state vector.
     */
    State<T>                                state() const;

    /**
     * @brief Check if the solver is currently positioned at an event.
     * @return True if an event has triggered at the current time.
     */
    bool                                    at_event() const;

    /**
     * @brief Print the current solver state to stdout.
     *
     * Includes event information in addition to the base state output.
     *
     * @param prec Number of decimal places for floating-point output.
     */
    void                                    show_state(int prec=8) const;

    // MODIFIERS

    /**
     * @brief Advance the solver until an event is detected.
     *
     * Repeatedly calls advance() until an event triggers or the solver
     * stops (due to error or reaching a dead state).
     *
     * @return True if an event was reached, false if solver stopped for other reasons.
     */
    bool                                    advance_to_event();

    /**
     * @brief Begin collecting dense output interpolation data.
     *
     * After calling this method, the solver builds a chain of interpolators
     * that can be queried via interpolator() to get the solution at any time
     * within the integration history.
     */
    void                                    start_interpolation();

    /**
     * @brief Stop collecting dense output interpolation data.
     *
     * Clears the interpolator chain and stops recording interpolation data.
     */
    void                                    stop_interpolation();

    RichSolver() = delete;

protected:

    /**
     * @brief Protected constructor for derived classes.
     * @param events Vector of event pointers to monitor during integration.
     * @see SOLVER_CONSTRUCTOR macro for other parameter details.
     */
    RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events);

    DEFAULT_RULE_OF_FOUR(RichSolver)

    ~RichSolver() = default;

    /// @brief Reset implementation hook. Resets events and stops interpolation.
    INLINE void reset_impl();

    /**
     * @brief Re-adjustment hook for state changes at events.
     * @param new_vector New state vector values (size Nsys).
     */
    INLINE void re_adjust_impl(const T* new_vector);

private:

    //================= STATIC OVERRIDES ======================
    /// @brief Time accessor that accounts for events.
    inline const T&     t_impl() const;

    /// @brief Vector accessor returning exposed state (masked or pre-mask based on visibility).
    inline const T*     vector_impl() const;

    /// @brief Advance implementation with event detection.
    bool                adv_impl();

    /// @brief Args update that also updates event arguments.
    inline void         set_args_impl(const T* new_args);
    //=========================================================

    /// @brief Add an interpolant to the linked interpolator chain.
    void            add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant);

    /// @brief Check if an event requires restarting integration.
    inline bool     requires_new_start() const;

    /// @brief Check if the current state equals the new computed state.
    inline bool     equiv_states() const;

    /// @brief Collection of events being monitored.
    EventCollection<T>                      _events;

    /// @brief Linked chain of interpolators for dense output.
    PolyWrapper<LinkedInterpolator<T, N>>   _cli;

    /// @brief Flag indicating if interpolation data is being collected.
    bool                                    _interp_data = false;

};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline const T& RichSolver<Derived, T, N, SP, RhsType, JacType>::t_impl() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::t_impl();
        }else{
            return event.state()->t();
        }
    }else{
        return Base::t_impl();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline const T* RichSolver<Derived, T, N, SP, RhsType, JacType>::vector_impl() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::vector_impl();
        }else{
            return _events.state(*ev_idx).exposed().vector();
        }
    }else{
        return Base::vector_impl();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline View1D<T, N> RichSolver<Derived, T, N, SP, RhsType, JacType>::true_vector() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return View1D<T, N>(Base::vector_impl(), this->Nsys());
        }else{
            return View1D<T, N>(_events.state(*ev_idx).True().vector(), this->Nsys());
        }
    }else{
        return View1D<T, N>(Base::vector_impl(), this->Nsys());
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
EventView<T> RichSolver<Derived, T, N, SP, RhsType, JacType>::current_events() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return EventView<T>(nullptr, nullptr, 0);
        }else{
            return _events.event_view();
        }
    }else{
        return EventView<T>(nullptr, nullptr, 0);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const EventCollection<T>& RichSolver<Derived, T, N, SP, RhsType, JacType>::event_col() const{
    return _events;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const Interpolator<T, N>* RichSolver<Derived, T, N, SP, RhsType, JacType>::interpolator() const{
    return _cli.ptr();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::is_interpolating() const{
    return _interp_data;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
State<T> RichSolver<Derived, T, N, SP, RhsType, JacType>::state() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::state();
        }else{
            return _events.state(*ev_idx).exposed();
        }
    }else{
        return Base::state();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::at_event() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        return static_cast<bool>(Base::t_impl() * this->direction() >= event.state()->t()*this->direction());
    }else{
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::show_state(int prec) const{
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message(), this->current_events()).show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RichSolver<Derived, T, N, SP, RhsType, JacType>::RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events) : Base(ARGS), _events(events){
    _events.setup(t0, this->args().data(), this->args().size(), this->Nsys(), this->direction());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::adv_impl(){

    if (this->requires_new_start()){
        this->re_adjust(this->true_vector().data());
    }

    if (this->equiv_states()){
        if (Base::adv_impl()){
            State<T> old_state = this->old_state();
            State<T> new_state = this->new_state();
            _events.detect_all_between(old_state, new_state, interp_func<Derived, T, N, SP, RhsType, JacType>, this);
            if (_interp_data){
                std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(0, -1);
                if (const EventState<T>* ev = _events.canon_state()){
                    r->adjust_end(ev->t());
                }
                this->add_interpolant(std::move(r));
            }
        }else{
            return false;
        }
    }else if (!Base::is_at_new_state()){

        bool event_waiting = false;
        if (const size_t* ev_idx = _events.begin()){
            const Event<T>& event = _events.event(*ev_idx);
            event_waiting = (Base::t_impl() * this->direction() < event.state()->t()*this->direction());
        }

        if (Base::adv_impl()){
            if (const size_t* idx_ptr = _events.begin()){
                const Event<T>& event = _events.event(*idx_ptr);
                if (!event_waiting && Base::t_impl() * this->direction() >= event.state()->t()*this->direction()){
                    _events.next_result();
                }
            }else{
                _events.next_result();
            }
        }else{
            return false;
        }
    }else{
        _events.next_result();
    }

    if (_events.begin()){
        if (_interp_data && this->requires_new_start()){
            if (!_events.canon_event()->hides_mask()){
                auto r = std::unique_ptr<Interpolator<T, N>>(new LocalInterpolator<T, N>(this->t(), this->true_vector().data(), this->Nsys()));
                _cli->adjust_end(this->t());
                this->add_interpolant(std::move(r));
            }
        }
    }

    if (_interp_data){
        _cli->close_end();
    }

    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::advance_to_event(){
    if (_events.size() == 0){
        return false;
    }
    do {
        if (!this->advance()){
            return false;
        }
    }while (!this->at_event());

    return true;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::start_interpolation(){
    if (!_interp_data){
        _interp_data = true;

        if (this->equiv_states()){
            _cli.take_ownership(new LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys()));
        }
        else{
            int bdr1 = 1;

            if (const size_t* ev_idx = _events.begin()){
                const T& t_event = _events.state(*ev_idx).t();
                if ((Base::t_impl() * this->direction() < t_event*this->direction()) && _events.canon_event() && (t_event == _events.canon_state()->t()) && _events.canon_event()->hides_mask()){
                    _cli.take_ownership(new LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys()));
                    bdr1 = -1;
                }
            }


            std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(bdr1, -1);
            r->adjust_start(this->t());

            if (bdr1 == 1){
                _cli.take_ownership(new LinkedInterpolator<T, N>(r.get()));
            }
            else{
                _cli->expand_by_owning(std::move(r));
            }
        }

    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::stop_interpolation(){
    _cli.take_ownership(nullptr);
    _interp_data = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    _events.reset();
    stop_interpolation();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
    Base::re_adjust_impl(new_vector);
    if (_interp_data){
        _cli->adjust_end(this->t());
    }
}

// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void RichSolver<Derived, T, N, SP, RhsType, JacType>::set_args_impl(const T* new_args){
    Base::set_args_impl(new_args);
    _events.set_args(new_args, this->args().size());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant){
    if (_cli->last_interpolant().interval().is_point() && interpolant->interval().start_bdr() == 0){
        interpolant->close_start();
    }
    _cli->expand_by_owning(std::move(interpolant));
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline bool RichSolver<Derived, T, N, SP, RhsType, JacType>::requires_new_start() const{
    return _events.canon_event() && (_events.canon_state()->t() == this->t());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline bool RichSolver<Derived, T, N, SP, RhsType, JacType>::equiv_states() const{
    return this->t_impl() == Base::t_new();
}

// ============================================================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void interp_func(T* res, const T& t, const void* obj){
    const auto* solver = reinterpret_cast<const RichSolver<Derived, T, N, SP, RhsType, JacType>*>(obj);
    solver->interp_impl(res, t);
}



/**
 * @brief Type alias that selects between BaseSolver and RichSolver based on policy.
 *
 * If the solver policy is RichStatic or RichVirtual, this resolves to RichSolver
 * (with event detection and dense output). Otherwise, it resolves to BaseSolver.
 *
 * @tparam Derived The derived solver class (CRTP pattern).
 * @tparam T       Scalar type for computations.
 * @tparam N       System size at compile time (0 for runtime-sized).
 * @tparam SP      Solver policy controlling which base class is selected.
 * @tparam RhsType Type of the right-hand side function.
 * @tparam JacType Type of the Jacobian function.
 */
template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
using BaseDispatcher = std::conditional_t<(SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual), RichSolver<Derived, T, N, SP, RhsType, JacType>, BaseSolver<Derived, T, N, SP, RhsType, JacType>>;

}


#endif


/*
TODO:

Check interpolation is working properly with advance_until, events etc.
simplify the code/logic in t_impl, vector_impl, (maybe using common code with lambdas)
add advance_until(t, worker_lambda) in BaseSolver.
*/