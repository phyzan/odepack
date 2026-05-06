#ifndef RICH_BASE_HPP
#define RICH_BASE_HPP

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
 * @tparam JacPolicy Type of the Jacobian function (or nullptr_t if not provided).
 *
 * @note This class uses static polymorphism (CRTP).
 *       The solver policy must be RichStatic or RichVirtual for this class to be used.
 */
template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
class RichSolver : public BaseSolver<Derived, T, N, SP, OdeType>{

    using Base = BaseSolver<Derived, T, N, SP, OdeType>;

    friend Base; // So that Base can access specific private methods for static override

public:

    // ACCESSORS
    /**
     * @brief Get a reference to the event collection.
     * @return Const reference to the internal EventCollection.
     */
    const EventCollection<T>&   event_col() const;

    bool                        at_event(int event_idx = -1) const;

    EventState<T>               current_event() const{
        if (this->at_event()){
            return EventState<T>{.event = &events.event(current_idx), .idx = current_idx, .is_masked = is_at_canon_event, .active = true};
        } else {
            return EventState<T>{.active = false};
        }
    }

    /**
     * @brief Get the index of a named event
     * @param name Name of the event to find.
     * @return Integer index of the event with the given name, or -1 if not found.
     */
    int                         event_idx(const std::string& name) const;

    std::vector<size_t>         toEventIdx(const std::vector<std::string>& event_names) const;

    /**
     * @brief Print the current solver state to stdout.
     *
     * Includes event information in addition to the base state output.
     *
     * @param prec Number of decimal places for floating-point output.
     */
    void                        show_state(int prec=8) const;

    // MODIFIERS

    /**
     * @brief Advance the solver until an event is detected.
     *
     * Repeatedly calls advance() until an event triggers or the solver
     * stops (due to error or reaching a dead state).
     *
     * @param event_idx Optional indices of specific events to advance to. If empty (default), advances to the next event regardless of type.
     * @return True if an event was reached, false if solver stopped for other reasons.
     */
    bool                        advance_to_event(const std::vector<size_t>& event_idx = {});

    /// @brief Same as advance_to_event but with a maximum time limit.
    bool                        advance_to_event(const T& tmax, const std::vector<size_t>& event_idx = {});

    bool                        advance_to_event(const std::vector<std::string>& event_names);

    bool                        advance_to_event(const T& tmax, const std::vector<std::string>& event_names);
    /// @brief Check if the solver is currently at a canon event.
    bool                        at_canon_event() const;

    /// @brief Reset implementation hook. Resets events and stops interpolation.
    void                        Reset();

    RichSolver() = delete;

protected:

    /**
     * @brief Protected constructor for derived classes.
     * @param events Vector of event pointers to monitor during integration.
     * @see SOLVER_CONSTRUCTOR macro for other parameter details.
     */
    RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events = {});

    DEFAULT_RULE_OF_FOUR(RichSolver)

    ~RichSolver() = default;

    /// @brief Advance implementation with event detection.
    template<typename... Args>
    bool        adv_impl(Args&&... args);

    /**
     * @brief Re-adjustment hook for state changes at events.
     * @param new_vector New state vector values (size Nsys).
     */
    void ReAdjust(const T* new_vector);

    struct Accessor : Derived {
        
        static void call_ReAdjust(Derived& self, const T* new_vector){
            constexpr auto fn = &Accessor::ReAdjust;
            (self.*fn)(new_vector);
        }
    };


    //================= STATIC OVERRIDES ======================

    bool    RequestTimeFloor(T& out) {
        // no need to call Base::RequestTimeFloor, the Base class does not request it.
        detection_idx = -1; // reset detection index at the start of a new detection round
        // do not also set is_at_event to false, as the adv_impl might fail and the step should remain in the same state.
        if ((is_event_waiting = events.detect_all_between(this->old_state(), this->new_state(), [this](T* out, const T& t){
            this->interp_impl(out, t);
        }))){
            // is_event_waiting has been set to true, preparing the push_event_queue for the first event
            if (Base::RequestTimeFloor(out)){
                out = this->minimum_time(out, events.get_time(0));
            } else {
                out = events.get_time(0);
            }
            return true;
        } else {
            return false;
        }
    }

    /// @brief Args update that also updates event arguments.
    void        set_args_impl(const T* new_args);
    //=========================================================

private:

    /// @brief Returns true if the queue was successfully advanced to the next event, updating flags, and false nothing happened.
    bool push_event_queue(){
        if (is_event_waiting && events.get_time(size_t(detection_idx+1)) == this->t()){
            current_idx = events.get_event_idx(size_t(++detection_idx));
            // determine if this one is a canon event
            if (const MaskedState<T>* ms = events.masked_state()){
                if ((is_at_canon_event = static_cast<bool>(ms->idx == current_idx)) && !events.event(current_idx).hides_mask()){
                    Accessor::call_ReAdjust(*THIS, ms->masked_vector.data());
                }
            }
            // determine if there is another event after this one
            is_event_waiting = size_t(detection_idx) < events.detection_size() - 1;
            is_at_event = true;
            return true;
        } else {
            return false;
        }
    }

    /// @brief Collection of events being monitored.
    EventCollection<T>                      events;
    size_t                                  current_idx = 0; //index of the currently triggered event. If not at an event, this should not be accessed.
    int                                     detection_idx = -1; //index of the currently triggered event in the detection order. At the start of an event detection this is -1, at the end of the iteration it is equal to the number of detections - 1.
    bool                                    is_at_event = false;
    bool                                    is_at_canon_event = false;
    bool                                    is_event_waiting = false;

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
 * @tparam JacPolicy Type of the Jacobian function.
 */
template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
using BaseDispatcher = std::conditional_t<(SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual), RichSolver<Derived, T, N, SP, OdeType>, BaseSolver<Derived, T, N, SP, OdeType>>;

} // namespace ode


#endif // RICH_BASE_HPP
