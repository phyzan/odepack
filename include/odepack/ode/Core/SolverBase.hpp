#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

/**
 * @file solverbase.hpp
 * @brief Base class template for ODE solvers using CRTP pattern.
 *
 * This file defines the BaseSolver template class, which provides the core
 * functionality for adaptive-step ODE integration. Derived solver classes
 * (e.g., RK45, DOP853) inherit from BaseSolver using the Curiously Recurring
 * Template Pattern (CRTP) to enable static polymorphism and avoid virtual
 * function overhead.
 *
 * The solver maintains internal state including current/previous solution
 * vectors, time values, step sizes, and integration status. It supports
 * forward and backward integration, adaptive step size control, dense
 * output interpolation, and event detection.
 */

#include "../Tools.hpp"
#include "VirtualBase.hpp"


#define MAIN_DEFAULT_CONSTRUCTOR(T) OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1

#define MAIN_CONSTRUCTOR(T) OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir

#define SOLVER_CONSTRUCTOR(T) OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir

#define ODE_CONSTRUCTOR(T) MAIN_DEFAULT_CONSTRUCTOR(T), EventList<T> events={}, Integrator method = Integrator::RK45


// For non-template member functions: use pointer-to-member (standard-compliant)
#define ODEPACK_ACCESSOR(NAME) \
    template<typename... Args> \
    static auto call_##NAME(Derived& self, Args&&... args){ \
        constexpr auto fn = &Accessor::NAME; \
        return (self.*fn)(std::forward<Args>(args)...); \
    } \
    template<typename... Args> \
    static auto call_##NAME(const Derived& self, Args&&... args){ \
        constexpr auto fn = &Accessor::NAME; \
        return (self.*fn)(std::forward<Args>(args)...); \
    } \

// For template member functions: use pointer-to-member with explicit template args
#define ODEPACK_ACCESSOR_TEMPLATE(NAME) \
    template<typename... Args> \
    static auto call_##NAME(Derived& self, Args&&... args){ \
        auto fn = &Accessor::template NAME<Args...>; \
        return (self.*fn)(std::forward<Args>(args)...); \
    } \
    template<typename... Args> \
    static auto call_##NAME(const Derived& self, Args&&... args){ \
        auto fn = &Accessor::template NAME<Args...>; \
        return (self.*fn)(std::forward<Args>(args)...); \
    } \

#define ODEPACK_CALL_DERIVED(NAME, ...) Accessor::call_##NAME(*THIS, __VA_ARGS__)


namespace ode{
/**
 * @brief Base class for adaptive-step ODE solvers using CRTP.
 *
 * BaseSolver provides the common infrastructure for ODE integration including
 * state management, step size adaptation, interpolation, and event detection.
 * Derived classes must implement the core stepping algorithm via static
 * method overrides.
 *
 * @tparam Derived The derived solver class (CRTP pattern). Must define:
 *                 - static constexpr bool IS_IMPLICIT
 *                 - static constexpr int ERR_EST_ORDER
 *                 - adapt_impl(), interp_impl(), local_interp()
 * @tparam T       Scalar type for computations (e.g., double, float).
 * @tparam N       System size at compile time. Use 0 for runtime-sized systems.
 * @tparam SP      Solver policy controlling behavior (see SolverPolicy enum).
 *
 * @note This class uses static polymorphism (CRTP). The derived class passes itself as the
 *       first template parameter.
 */

namespace detail{

// Whether a solver's scratch buffers can live in automatic storage. That needs two things:
// a system size known at compile time, and a scalar cheap enough that creating the buffer is
// free. A type like mpfr::mpreal is neither trivially constructible nor trivially copyable -
// every element owns a heap allocation - so a fresh stack array per access would construct and
// destroy nsys of them on every step. Those types use the persistent heap-backed form instead,
// which allocates once in the constructor and hands out references.
template<typename T, size_t N>
inline constexpr bool scratch_is_static = (N > 0) && std::is_trivially_copyable_v<T>
                                          && std::is_trivially_default_constructible_v<T>;

// Scratch storage. If a function requires s scratch buffer for a state vector,
// meaning that the size is (nsys+2), then the scratch buffer is allocated on the stack
// if the system size is known at compile time and the scalar type is trivially copyable and default constructible (e.g. double)
// Otherwise, it is allocated on the heap once in the constructor and reused (e.g. mpfr::mpreal or dynamic size system of any type).
template<typename T, size_t N>
using ScratchState = Array1D<T, (N > 0 ? N+2 : 0),
                           scratch_is_static<T, N> ? Allocation::Auto : Allocation::Heap>;

template<typename T, size_t N>
class StaticSolverScratch{

public:

    using DualType = ::ode::DualType<T, N, 1>;

    StaticSolverScratch(size_t nsys) {
        assert(nsys == N && "SolverScratchSpace: nsys must match template parameter N for fixed-size systems.");
    }

    ScratchState<T, N> state() const {return ScratchState<T, N>{};}
    Array1D<T, 4*N> four_state_cache() const {return Array1D<T, 4*N>{};}
    Array1D<T, N> ics_cache() const {return Array1D<T, N>{};}
    Array1D<DualType, 2*N> duals() const {return Array1D<DualType, 2*N>{};}
};


// Heap-backed scratch. Used for runtime-sized systems, and for any scalar that is not trivial
// to construct/copy. N is still carried so that DualType keeps the solver's derivative width.
template<typename T, size_t N>
class DynamicSolverScratch{

public:

    using DualType = ::ode::DualType<T, N, 1>;

    DynamicSolverScratch(size_t nsys) : state_(nsys+2), four_state_cache_(4*nsys), ics_cache_(nsys), duals_(2*nsys) {}

    ScratchState<T, N>& state() const {return state_;}
    Array1D<T>& four_state_cache() const {return four_state_cache_;}
    Array1D<T>& ics_cache() const {return ics_cache_;}
    Array1D<DualType>& duals() const {return duals_;}
private:
    mutable ScratchState<T, N> state_; // for trying the next step
    mutable Array1D<T> four_state_cache_; // for approx jac and auto step
    mutable Array1D<T> ics_cache_; // for trying the next step with modified ICs
    mutable Array1D<DualType> duals_; // for autodiff when JP==JacPolicy::Autodiff
};

template<typename T, size_t N>
using SolverScratchSpace = std::conditional_t<scratch_is_static<T, N>,
                                              StaticSolverScratch<T, N>,
                                              DynamicSolverScratch<T, N>>;

} // ode::detail


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
class BaseSolver : public traits::SolverVirtualTypeTraits<Derived, T, N, SP>::type {

    using CloneType = traits::SolverCloneType<Derived, T, N, SP>;

public:

    using Base = typename traits::SolverVirtualTypeTraits<Derived, T, N, SP>::type;
    using DualType = ::ode::DualType<T, N, 1>;
    static constexpr JacPolicy JP = getJacPolicy<T, N, OdeType>();
    

    using value_type = T;
    using ode_type = OdeType;
    static constexpr size_t NSYS = N;
    static constexpr SolverPolicy Policy = SP;

    
    BaseSolver() = delete;

    // ODE PROPERTIES

    /**
     * @brief Evaluate the right-hand side of the ODE system.
     * @param[out] dq_dt Output array for the derivative dq/dt (size Nsys).
     * @param[in]  t     Current time.
     * @param[in]  q     Current state vector (size Nsys).
    */
    void                 Rhs(T* dq_dt, const T& t, const T* q) const;

    /**
     * @brief Compute the Jacobian matrix of the ODE system.
     * 
     * Uses the exact Jacobian if provided, otherwise falls back to
     * finite difference approximation via jac_approx().
     *
     * @param[out] jm Output array for Jacobian in column-major order (size Nsys x Nsys).
     * @param[in]  t  Current time.
     * @param[in]  q  Current state vector (size Nsys).
     * @param[in]  dt Optional step sizes for finite difference (size Nsys). If nullptr,
     *                step sizes are computed automatically.
    */
    void                 Jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const;

    /**
     * @brief Approximate the Jacobian using central finite differences.
     * @param[out] out  Output array for Jacobian in column-major order (size Nsys x Nsys).
     * @param[in]  t  Current time.
     * @param[in]  q  Current state vector (size Nsys).
     * @param[in]  dt Step sizes for each component (size Nsys). If nullptr, computed automatically.
    */
    void                 jac_approx(T* out, const T& t, const T* q, const T* dt) const;

    /**
     * @brief Get a matrix view of a Jacobian array.
     *
     * Returns a 2D view that allows element access via matrix(i, j) syntax.
     *
     * @param[in] j Pointer to Jacobian data in column-major order.
     * @return MutView object for convenient matrix element access.
    */
    MutView<T, Layout::F, N, N> jac_view(T* j) const;

    // ACCESSORS

    /// @brief Get the current time value.
    const T&            t() const;

    /// @brief Get the time value of the newest computed step.
    const T&            t_new() const;

    /// @brief Get the time value from the previous accepted step.
    const T&            t_old() const;

    /// @brief Get a view of the current state vector.
    View1D<T, N>        vector() const;

    /// @brief Get a view of the state vector from the newest computed step.
    View1D<T, N>        vector_new() const;

    /// @brief Get a view of the state vector from the previous accepted step.
    View1D<T, N>        vector_old() const;

    /// @brief Get a State object representing the initial conditions.
    State<T>            ics() const;

    /// @brief Get a State object representing the most recent computed step.
    State<T>            new_state() const;

    /// @brief Get a State object representing the previous accepted step.
    State<T>            old_state() const;

    /// @brief Get the current step size (absolute value).
    const T&            stepsize() const;

    /// @brief Get the integration direction (+1 forward, -1 backward).
    int                 direction() const;

    /// @brief Get the relative tolerance for error control.
    const T&            rtol() const;

    /// @brief Get the absolute tolerance for error control.
    const T&            atol() const;

    /// @brief Get the minimum allowed step size.
    const T&            min_step() const;

    /// @brief Get the maximum allowed step size.
    const T&            max_step() const;

    /// @brief Get the number of equations in the ODE system.
    constexpr size_t    nsys() const {if constexpr (N > 0) {return N;} else {return nsys_;}}
    
    /// @brief Get the number of successful integration steps taken.
    size_t              step_count() const;

    /// @brief Check if the solver is currently running (not paused or dead).
    bool                is_running() const;

    /// @brief Check if the solver has permanently terminated.
    bool                is_dead() const;

    /// @brief Check if the solution has diverged (contains inf/nan).
    bool                diverges() const;

    /// @brief Get the current status message.
    const std::string&  status() const;

    /**
     * @brief Print the current solver state to stdout.
     * @param prec Number of decimal places for floating-point output.
     */
    void                show_state(int prec=8) const;

    /**
     * @brief Validate proposed initial conditions.
     * @param t0 Initial time.
     * @param q0 Initial state vector.
     * @return True if the ICs are valid (finite values, finite RHS evaluation).
    */
    bool                validate_ics(T t0, const T* q0) const;

    /**
     * @brief Interpolate the solution at a time within the last step interval.
     * @param[out] out Output array for interpolated state (size Nsys).
     * @param[in]  t      Time to interpolate at (must be in [t_old, t_new]).
     * @throws std::runtime_error If t is outside the valid interpolation range.
    */
    void                interp(T* out, const T& t) const;

    /**
     * @brief Get the number of RHS function evaluations performed so far.
     * @return Total count of RHS evaluations. User calls to Rhs() do NOT increment this counter.
    */
    size_t              rhs_eval_count() const;

    /**
     * @brief Get the number of Jacobian evaluations performed so far.
     * @return Total count of Jacobian evaluations. User calls to Jac() do NOT increment this counter.
    */
    size_t              jac_eval_count() const;

    /**
     * @brief Compute an appropriate initial step size.
     * @param t Initial time.
     * @param q Initial state vector.
     * @return Suggested step size based on local scales and tolerances.
     */
    T                   auto_step(T t, const T* q) const;

    /// @brief Compute an appropriate step size from the current state.
    T                   auto_step() const;

    /**
     * @brief Create a dynamically allocated copy of this solver.
     * @return Pointer to a new solver instance. Caller owns the memory.
     */
    std::unique_ptr<CloneType> clone() const;

    // MODIFIERS

    /**
     * @brief Advance the solver by one integration step.
     * @return True if the step was successful, false if paused or dead.
     */
    bool                advance();

    bool                advance_until(const T& time);

    /**
     * @brief Integrate until the specified time is reached.
     * @param time Target time to integrate to.
     * @param observer Callable function(t, q_ptr, t_ptr) ->bool that is called at each successfull step until "time" is reached.
     *     observer arguments:
     *          t: solver's current integration time.
     *          q_ptr: solver's current state vector
     *          t_ptr: if positioned at an extra_step, pointer to the t-value in the array. Otherwise nullptr. If no extra_steps passed,
     *              then t_ptr is always nullptr except for the last call (if successfull), in which case t_ptr = &time.
     * the observer should return true to continue integration, or false to stop advancing anymore.
     * @param extra_steps Optional array of additional time points to observe (must be in the same direction and within the integration range). Observer will be called at these points as well.
     * @return True if integration succeeded reaching the target time, false if solver stopped early.
     */
    template<OptionalObserver<T> Callable, typename ArrayType = EmptyArr<T>>
    bool                advance_until(const T& time, const Callable& observer, const ArrayType& extra_steps = EmptyArr<T>());

    bool                observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer, View1D<T> extra_steps);

    bool                observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer);

    /// @brief observer(t, q_ptr, t_ptr) -> bool
    template<OptionalObserver<T> Callable = std::nullptr_t>
    BoxedInterp<T, N>   interpolate_until(const T& time, const Callable& observer = nullptr);

    BoxedInterp<T, N>   interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer = [](const auto&, const auto*, const auto*){return true;});

    /**
     * @brief Advance the solver by a specified time interval (along the integration direction).
     * @param interval Time interval to advance by (must be positive).
     * @return True if the interval was successfully integrated, false if paused or dead.
     * @note This is a convenience method equivalent to advance_until(t() + interval*direction()).
     *      The is not a single step advance; the solver will take as many steps as needed to reach the target time,
     *      and use interpolation to end exactly at the target time.
    */
    bool                advance_by(T interval);

    /**
     * @brief Set new initial conditions via a setter function.
     *
     * @tparam Setter Function type with signature: void(T* q) that fills the state vector.
     * @param t0      New initial time.
     * @param func    Function that writes the initial state to the provided pointer, whose data are the solver's initial conditions, NOT the current state.
     * @param stepsize Initial step size (0 = auto-compute).
     */
    template<typename Setter>
    auto                apply_ics_setter(T t0, Setter&& func, T stepsize = 0);

    /// @brief Same as apply_ics_setter, but the setter function receives the current state as an argument to modify. Then e.g. new_vector[i] += 1 would increment the i-th component of the current state by 1 before restarting, and works as expected.
    template<typename Setter>
    auto                restart_from_modified_state(T t0, Setter&& func, T stepsize = 0);
    /**
     * @brief Set new initial conditions without reallocating memory.
     * @param t0      New initial time.
     * @param y0      New initial state vector (size Nsys).
     * @param stepsize Initial step size (0 = auto-compute).
     * @param direction Integration direction for the new ICs (+1 forward, -1 backward, 0 for unchanged).
     * @return True if ICs were valid and set successfully. Otherwise returns false and stops the solver. Simply call resume() to continue.
     * @throws std::runtime_error If stepsize is negative.
     */
    bool                set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0);

    /**
     * @brief Pause the solver (can be resumed later).
     * @param text Optional message describing why the solver was stopped.
     */
    void                stop(const std::string& text = "");

    /**
     * @brief Permanently terminate the solver (cannot be resumed).
     * @param text Optional message describing why the solver was killed.
     */
    void                kill(const std::string& text = "");

    /**
     * @brief Resume a paused solver.
     * @return True if resumed successfully, false if solver is dead.
     */
    bool                resume();

    /// @brief Reset implementation hook. Derived should call base first.
    void                Reset();

    /**
     * @brief Create an interpolator for dense output between two boundaries.
     * @param bdr1 First boundary index.
     * @param bdr2 Second boundary index.
     * @return Unique pointer to an interpolator object.
     * @note Must be implemented by derived class.
     */
    VirtualInterp<T, N> state_interpolator(int bdr1, int bdr2) const;

    // VIRTUAL INTERFACE ALIASES (inline overrides to avoid virtual calls)
    // Accessors
    void                get_rhs(T* dq_dt, const T& t, const T* q) const { Rhs(dq_dt, t, q); }
    void                get_jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const { Jac(jm, t, q, dt); }
    void                get_jac_approx(T* j, const T& t, const T* q, const T* dt) const { jac_approx(j, t, q, dt); }
    const T&            get_time() const { return t(); }
    const T&            get_new_time() const { return t_new(); }
    const T&            get_old_time() const { return t_old(); }
    View1D<T, N>        get_vector() const { return vector(); }
    View1D<T, N>        get_new_vector() const { return vector_new(); }
    View1D<T, N>        get_old_vector() const { return vector_old(); }
    State<T>            get_ics() const { return ics(); }
    State<T>            get_new_state() const { return new_state(); }
    State<T>            get_old_state() const { return old_state(); }
    const T&            get_stepsize() const { return stepsize(); }
    int                 get_direction() const { return direction(); }
    const T&            get_rtol() const { return rtol(); }
    const T&            get_atol() const { return atol(); }
    const T&            get_min_step() const { return min_step(); }
    const T&            get_max_step() const { return max_step(); }
    size_t              get_nsys() const { return nsys(); }
    size_t              get_step_count() const { return step_count(); }
    bool                get_is_running() const { return is_running(); }
    bool                get_is_dead() const { return is_dead(); }
    bool                get_diverges() const { return diverges(); }
    const std::string&  get_status() const { return status(); }
    bool                get_validate_ics(T t0, const T* q0) const { return validate_ics(t0, q0); }
    Integrator          get_method() const { return method(); }
    void                get_interp(T* result, const T& t) const { interp(result, t); }
    size_t              get_rhs_eval_count() const { return rhs_eval_count_; }
    size_t              get_jac_eval_count() const { return jac_eval_count_; }
    VirtualInterp<T, N> get_state_interpolator(int bdr1, int bdr2) const { return state_interpolator(bdr1, bdr2); }
    T                   get_auto_step(T t, const T* q) const { return auto_step(t, q); }
    T                   get_auto_step() const { return auto_step(); }
    
    // Modifiers
    bool                do_advance() { return advance(); }
    bool                do_advance_by(T interval) { return advance_by(interval); }
    bool                do_advance_until(const T& time) { return advance_until(time); }
    bool                do_observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer) { return observe_until(time, observer); }
    bool                do_observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer, View1D<T> extra_steps) { return observe_until(time, observer, extra_steps); }
    BoxedInterp<T, N>   do_interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer = [](const auto&, const auto*, const auto*){return true;}) { return interp_until(time, observer); }
    void                do_reset() { THIS->Reset(); }
    bool                do_resume() { return resume(); }
    void                do_stop(const std::string& text = "") { stop(text); }
    void                do_kill(const std::string& text = "") { kill(text); }
    bool                do_set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0) { return set_ics(t0, y0, stepsize, direction); }

    // =================== STATIC OVERRIDES (NECESSARY) ===============================
    // Derived classes MUST implement these methods / attributes.

    Integrator method() const {
        return THIS->method();
    }

protected:

    using MainSolverType = BaseSolver;


    /// @brief Whether the method is implicit (must be defined in Derived).
    static constexpr bool       IS_IMPLICIT = Derived::IS_IMPLICIT;
    /// @brief Order of the error estimator (must be defined in Derived).
    static constexpr int        ERR_EST_ORDER = Derived::ERR_EST_ORDER;

    /**
     * @brief Perform one adaptive integration step.
     * @param[out] state Output array for the new state [t, h, q...] (size Nsys+2).
     * @note Must be implemented by derived class.
     */
    StepResult              adapt_impl(T* state, const T* old_state);

    /**
     * @brief Interpolate solution at time t using method-specific interpolation.
     * @param[out] result Output array for interpolated state (size Nsys).
     * @param[in]  t      Time to interpolate at.
     * @note Must be implemented by derived class.
     */
    void                    interp_impl(T* result, const T& t) const;

    auto                    local_interp() const;

    // ================================================================================

    // ========================= STATIC OVERRIDES (OPTIONAL) ==========================
    // Derived classes MAY override these methods. Call base implementation first.

    /// @brief Advance implementation. If t_lim is nullptr, advance normally. Otherwise, if the implementation yields a time beyond t_lim, it should only cache the step information (e.g. counters, or event or state register etc) since the solver is "not to go there yet" and only apply them once Adv_Impl is called with t_lim beyond the implementation's predicted step.
    template<typename... Args>
    bool        Adv_Impl(Args&&... args);

    constexpr bool    RequestTimeFloor(T& /*out*/) {
        return false;
    }

    /**
    @brief Re-adjustment hook right before new_state modification. Derived should call base first.
    @param new_vector New state vector values (size Nsys).

    @note Nothing has changed yet when this is called; it's a chance to update any internal data before the state is modified. The new state will be set to (t(), stepsize(), new_vector),
    where t() is the true current time, which might lie between old_state and new_state (e.g. if an event occurred).
    */
    void    ReAdjust(const T* new_vector);

    /**
     * @brief Validate initial conditions implementation.
     * @param t0 Initial time.
     * @param q0 Initial state.
     * @return True if ICs are valid.
     * @note Derived should call base first, then add additional checks.
     */
    bool    validate_ics_impl(T t0, const T* q0) const;

    // ================================================================================


    struct Accessor : Derived {

        ODEPACK_ACCESSOR(adapt_impl)
        ODEPACK_ACCESSOR(interp_impl)
        ODEPACK_ACCESSOR(local_interp)
        ODEPACK_ACCESSOR_TEMPLATE(Adv_Impl)  // Template member function
        ODEPACK_ACCESSOR(RequestTimeFloor)
        ODEPACK_ACCESSOR(ReAdjust)
        ODEPACK_ACCESSOR(validate_ics_impl)
    };


    // =========================== HELPER METHODS =====================================

    inline const OdeType& ode() const {return ode_;}

    /// @brief Same as this->Rhs, but increments the RHS evaluation counter.
    void        rhs(T* dq_dt, const T& t, const T* q) const;

    /// @brief Same as this->Jac, but increments the Jacobian evaluation counter.
    void        jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const;

    /// @brief Get pointer to the initial conditions state data.
    const T*    ics_ptr() const;

    /// @brief Get pointer to the current "true" state.
    const T*    true_state_ptr() const;

    /// @brief Get pointer to the previous "true" state.
    const T*    last_true_state_ptr() const;

    /// @brief Get pointer to the most recently computed state.
    const T*    new_state_ptr() const;

    /// @brief Get pointer to the previous accepted state.
    const T*    old_state_ptr() const;

    /// @brief Get pointer to the correct new state for interpolation
    const T*    interp_new_state_ptr() const;

    /// @brief Print a warning that the solver is paused.
    void        warn_paused() const;

    /// @brief Print a warning that the solver is dead.
    void        warn_dead() const;

    /// @brief Set the solver status message.
    void        set_message(const std::string& text);

    /// @brief Check if the current true state matches the new state.
    bool        is_at_new_state() const;


    void cerr(const std::string& message) const {
        #pragma omp critical
        {
            std::cerr << message << std::endl;
        }
    }

    template<typename U, typename... Args>
    T    minimum_time(const U& item, const Args&... args) const{
        if constexpr (sizeof...(Args) > 0){
            return minimum_time_helper(item, args...);
        } else {
            return item;
        }
    }

    // ================================================================================

    DEFAULT_RULE_OF_FOUR(BaseSolver)

    /**
     * @brief Protected constructor for derived classes.
     * @see SOLVER_CONSTRUCTOR macro for parameter details.
     */
    BaseSolver(SOLVER_CONSTRUCTOR(T));
    ~BaseSolver() = default;

    /// @brief Maximum step size increase factor per step.
    T                                   MAX_FACTOR = 10;

    /// @brief Safety factor for step size control (typically 0.8-0.95).
    T                                   SAFETY = T(9)/10;

    /// @brief Minimum step size decrease factor per step.
    T                                   MIN_FACTOR = T(2)/10;
    
    /// @brief Absolute minimum step size before solver terminates.
    T                                   MIN_STEP = 100*std::numeric_limits<T>::epsilon();

private:

    bool                    validate_it(StepResult result, const T* state);
    void                    set_state(const T& time, T* state);

    template<typename A, typename B, typename... Args>
    const T&    minimum_time_helper(const A& a, const B& b, Args&&... args) const{
        if constexpr (sizeof...(args) > 0){
            return minimum_time_helper(min_of(a, b), args...);
        }else{
            return min_of(a, b);
        }
    }

    template<typename A, typename B>
    const T& min_of(const A& a, const B& b) const{
        if (this->direction() == 1){
            return (a < b ? a : b);
        }else{
            return (a > b ? a : b);
        }
    }

    /// @brief Only use inside Adv_Impl (so that if the state here is updated, all derived classes are aware). Move the current state to a new time between the current time and the most recently adapted state. This is a lowlevel operation, so use carefully or the intended bahavior might break.
    void                    move_state(const T& time);

    template<typename Setter>
    auto                    priv_apply_ics_setter(T* ics, T t0, Setter&& func, T stepsize);

    
    detail::ScratchState<T, N> ics_state_, old_state_, new_state_, true_state_, interp_state_;
    T rtol_, atol_, min_step_, max_step_;
    detail::SolverScratchSpace<T, N> scratch_;
    OdeType                                             ode_;
    size_t                                              nsys_ = N;
    size_t                                              step_count_ = 0;
    mutable size_t                                      rhs_eval_count_ = 0;
    mutable size_t                                      jac_eval_count_ = 0;
    std::string                                         msg_ = "Running";
    int                                                 direction_ = 1;
    bool                                                is_dead_ = false;
    bool                                                diverges_ = false;
    bool                                                is_running_ = true;
    bool                                                use_new_state_ = true; //for interpolation purposes
    bool                                                is_at_new_state_ = true;
};


namespace traits{

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

} // namespace ode::traits


#define SolverTemplate template<typename T, size_t, SolverPolicy, hasRhsFunc<T>, typename>


} // namespace ode

#endif
