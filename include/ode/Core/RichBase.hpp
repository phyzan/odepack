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
 void interp_func(T* res, const T& t, const void* obj);

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
     View1D<T, N>                     true_vector() const;

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
     void reset_impl();

    /**
     * @brief Re-adjustment hook for state changes at events.
     * @param new_vector New state vector values (size Nsys).
     */
     void re_adjust_impl(const T* new_vector);

private:

    //================= STATIC OVERRIDES ======================
    /// @brief Time accessor that accounts for events.
     const T&     t_impl() const;

    /// @brief Vector accessor returning exposed state (masked or pre-mask based on visibility).
     const T*     vector_impl() const;

    /// @brief Advance implementation with event detection.
    bool                adv_impl();

    /// @brief Args update that also updates event arguments.
     void         set_args_impl(const T* new_args);
    //=========================================================

    /// @brief Add an interpolant to the linked interpolator chain.
    void            add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant);

    /// @brief Check if an event requires restarting integration.
     bool     requires_new_start() const;

    /// @brief Check if the current state equals the new computed state.
     bool     equiv_states() const;

    /// @brief Collection of events being monitored.
    EventCollection<T>                      _events;

    /// @brief Linked chain of interpolators for dense output.
    PolyWrapper<LinkedInterpolator<T, N>>   _cli;

    /// @brief Flag indicating if interpolation data is being collected.
    bool                                    _interp_data = false;

};


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

} // namespace ode


#endif // RICH_SOLVER_HPP


/*
TODO:

Check interpolation is working properly with advance_until, events etc.
simplify the code/logic in t_impl, vector_impl, (maybe using common code with lambdas)
add advance_until(t, worker_lambda) in BaseSolver.
*/