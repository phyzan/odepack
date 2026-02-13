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
#include "../SolverState.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T) OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}

#define MAIN_CONSTRUCTOR(T) OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args

#define SOLVER_CONSTRUCTOR(T) OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args

#define ODE_CONSTRUCTOR(T) MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events={}, const std::string& method="RK45"

#define ARGS ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args

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
 *                 - static constexpr const char* name
 *                 - static constexpr bool IS_IMPLICIT
 *                 - static constexpr int ERR_EST_ORDER
 *                 - adapt_impl(), interp_impl(), state_interpolator()
 * @tparam T       Scalar type for computations (e.g., double, float).
 * @tparam N       System size at compile time. Use 0 for runtime-sized systems.
 * @tparam SP      Solver policy controlling behavior (see SolverPolicy enum).
 *
 * @note This class uses static polymorphism (CRTP). The derived class passes itself as the
 *       first template parameter.
 */


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
class BaseSolver : public BaseInterface<T, N, SP>{

    using Base = BaseInterface<T, N, SP>;
    using Clone = SolverCloneType<Derived, T, N, SP>;

    static constexpr bool RUNTIME_JAC_TYPE = std::is_same_v<JacType, void>;
    static constexpr bool HAS_JAC = !RUNTIME_JAC_TYPE && !std::is_same_v<JacType, std::nullptr_t>;
    static_assert( !std::is_same_v<RhsType, std::nullptr_t> , "RHS function type cannot be nullptr");

    static constexpr auto VoidFunc = [](const T&, const T*)LAMBDA_INLINE{};

public:

    using Scalar = T;
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
     * @param[out] j  Output array for Jacobian in column-major order (size Nsys x Nsys).
     * @param[in]  t  Current time.
     * @param[in]  q  Current state vector (size Nsys).
     * @param[in]  dt Step sizes for each component (size Nsys). If nullptr, computed automatically.
    */
    void                 jac_approx(T* j, const T& t, const T* q, const T* dt) const;

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
    const T&             t() const;

    /// @brief Get the time value from the previous accepted step.
    const T&              t_old() const;

    /// @brief Get a view of the current state vector.
    View1D<T, N>         vector() const;

    /// @brief Get a view of the state vector from the previous accepted step.
    View1D<T, N>         vector_old() const;

    /// @brief Get the current step size (absolute value).
    const T&             stepsize() const;

    /// @brief Get the integration direction (+1 forward, -1 backward).
    int                  direction() const;

    /// @brief Get the relative tolerance for error control.
    const T&             rtol() const;

    /// @brief Get the absolute tolerance for error control.
    const T&             atol() const;

    /// @brief Get the minimum allowed step size.
    const T&             min_step() const;

    /// @brief Get the maximum allowed step size.
    const T&             max_step() const;

    /// @brief Get the additional arguments passed to the ODE function.
    const Array1D<T>&    args() const;

    /// @brief Get the number of equations in the ODE system.
    size_t               Nsys() const;

    /// @brief Get the number of successful integration steps taken.
    size_t               Nupdates() const;

    /// @brief Check if the solver is currently running (not paused or dead).
    bool                 is_running() const;

    /// @brief Check if the solver has permanently terminated.
    bool                 is_dead() const;

    /// @brief Check if the solution has diverged (contains inf/nan).
    bool                 diverges() const;

    /// @brief Get the current status message.
    const std::string&   message() const;

    /**
     * @brief Print the current solver state to stdout.
     * @param prec Number of decimal places for floating-point output.
     */
    void                        show_state(int prec=8) const;

    /// @brief Get a State object representing the most recent computed step.
    State<T>             new_state() const;

    /// @brief Get a State object representing the previous accepted step.
    State<T>             old_state() const;

    /// @brief Get a State object representing the current solver position.
    State<T>             state() const;

    /// @brief Get a State object representing the initial conditions.
    State<T>             ics() const;

    /**
     * @brief Validate proposed initial conditions.
     * @param t0 Initial time.
     * @param q0 Initial state vector.
     * @return True if the ICs are valid (finite values, finite RHS evaluation).
    */
    bool                        validate_ics(T t0, const T* q0) const;

    /// @brief Get the name of the integration method (e.g., "RK45").
    const std::string&          method() const;

    /**
     * @brief Interpolate the solution at a time within the last step interval.
     * @param[out] result Output array for interpolated state (size Nsys).
     * @param[in]  t      Time to interpolate at (must be in [t_old, t_new]).
     * @throws std::runtime_error If t is outside the valid interpolation range.
    */
    void                 interp(T* result, const T& t) const;

    /**
     * @brief Get the number of RHS function evaluations performed so far.
     * @return Total count of RHS evaluations. User calls to Rhs() do NOT increment this counter.
    */
    size_t               n_evals_rhs() const;

    /**
     * @brief Get the number of Jacobian evaluations performed so far.
     * @return Total count of Jacobian evaluations. User calls to Jac() do NOT increment this counter.
    */
    size_t               n_evals_jac() const;

    /**
     * @brief Compute an appropriate initial step size.
     * @param t Initial time.
     * @param q Initial state vector.
     * @return Suggested step size based on local scales and tolerances.
     */
    T                           auto_step(T t, const T* q) const;

    /// @brief Compute an appropriate step size from the current state.
    T                           auto_step() const;

    /**
     * @brief Create a dynamically allocated copy of this solver.
     * @return Pointer to a new solver instance. Caller owns the memory.
     */
    Clone*                      clone() const;

    // MODIFIERS

    /**
     * @brief Advance the solver by one integration step.
     * @return True if the step was successful, false if paused or dead.
     */
    bool                        advance();

    bool                        advance_until(T time);

    /**
     * @brief Integrate until the specified time is reached.
     * @param time Target time to integrate to.
     * @param observer Callable function(t, q_ptr) that is called at each successfull step until "time" is reached.
     * @return True if integration succeeded, false if solver stopped early.
     */
    template<typename Callable>
    bool                        advance_until(T time, Callable&& observer);

    /**
     * @brief Integrate until an objective function crosses zero (event detection).
     *
     * @tparam Callable Function type with signature: T(const T& t, const T* q, const T* args, const void* obj)
     * @param obj_fun  Objective function to monitor for zero crossing.
     * @param tol      Tolerance for root finding (bisection).
     * @param dir      Direction of crossing: +1 (increasing), -1 (decreasing), 0 (any).
     * @param observer Callable function(t, q_ptr) that is called at each successfull step until the requested zero crossing of the objective function is reached
     * @param worker   Optional preallocated array (size Nsys) for temporary storage.
     *                 Use if obj_fun calls solver methods that may overwrite internal arrays.
     * @return True if event was detected and solver positioned at crossing.
     */
    template<typename ObjFun, typename Callable>
    bool                        advance_until(ObjFun&& obj_fun, T tol, int dir=0, Callable&& observer = decltype(VoidFunc)(VoidFunc), T* worker = nullptr);

    bool                        observe_until(T time, std::function<void(const T&, const T*)> observer);

    /// @brief Reset the solver to its initial conditions.
    void                        reset();

    /**
     * @brief Set new initial conditions via a setter function.
     *
     * @tparam Setter Function type with signature: void(T* q) that fills the state vector.
     * @param t0      New initial time.
     * @param func    Function that writes the initial state to the provided pointer.
     * @param stepsize Initial step size (0 = auto-compute).
     */
    template<typename Setter>
    void                        apply_ics_setter(T t0, Setter&& func, T stepsize = 0);

    /**
     * @brief Set new initial conditions without reallocating memory.
     * @param t0      New initial time.
     * @param y0      New initial state vector (size Nsys).
     * @param stepsize Initial step size (0 = auto-compute).
     * @param direction Integration direction for the new ICs (+1 forward, -1 backward, 0 default).
     * @return True if ICs were valid and set successfully. Otherwise returns false and stops the solver. Simply call resume() to continue.
     * @throws std::runtime_error If stepsize is negative.
     */
    bool                        set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0);

    /**
     * @brief Pause the solver (can be resumed later).
     * @param text Optional message describing why the solver was stopped.
     */
    void                        stop(const std::string& text = "");

    /**
     * @brief Permanently terminate the solver (cannot be resumed).
     * @param text Optional message describing why the solver was killed.
     */
    void                        kill(const std::string& text = "");

    /**
     * @brief Resume a paused solver.
     * @return True if resumed successfully, false if solver is dead.
     */
    bool                        resume();

    /**
     * @brief Set the user object pointer passed to ODE callbacks.
     * @param obj Pointer to user data (must not be this solver instance).
     */
     void                 set_obj(const void* obj);

    /**
     * @brief Update the additional arguments passed to the ODE function.
     * @param new_args Pointer to new argument values (must match original size).
     */
     void                 set_args(const T* new_args);


protected:

    using MainSolverType = BaseSolver;
    // =================== STATIC OVERRIDES (NECESSARY) ===============================
    // Derived classes MUST implement these methods.

    /// @brief Name of the integration method (must be defined in Derived).
    static constexpr const char*    name = Derived::name;
    /// @brief Whether the method is implicit (must be defined in Derived).
    static constexpr bool           IS_IMPLICIT = Derived::IS_IMPLICIT;
    /// @brief Order of the error estimator (must be defined in Derived).
    static constexpr int            ERR_EST_ORDER = Derived::ERR_EST_ORDER;

    /**
     * @brief Create an interpolator for dense output between two boundaries.
     * @param bdr1 First boundary index.
     * @param bdr2 Second boundary index.
     * @return Unique pointer to an interpolator object.
     * @note Must be implemented by derived class.
     */
     VirtualInterp<T, N>  state_interpolator(int bdr1, int bdr2) const;

    /**
     * @brief Perform one adaptive integration step.
     * @param[out] state Output array for the new state [t, h, q...] (size Nsys+2).
     * @note Must be implemented by derived class.
     */
     void                 adapt_impl(T* state);

    /**
     * @brief Interpolate solution at time t using method-specific interpolation.
     * @param[out] result Output array for interpolated state (size Nsys).
     * @param[in]  t      Time to interpolate at.
     * @note Must be implemented by derived class.
     */
     void                 interp_impl(T* result, const T& t) const;
    // ================================================================================

    // ========================= STATIC OVERRIDES (OPTIONAL) ==========================
    // Derived classes MAY override these methods. Call base implementation first.


    /// @brief Reset implementation hook. Derived should call base first.
     void                 reset_impl();

    /// @brief Args update implementation hook. Derived should call base first.
     void                 set_args_impl(const T* new_args);

    /**
    @brief Re-adjustment hook right before new_state modification. Derived should call base first.
    @param new_vector New state vector values (size Nsys).

    @note Nothing has changed yet when this is called; it's a chance to update any internal data before the state is modified. The new state will be set to (t(), stepsize(), new_vector),
    where t() is the true current time, which might lie between old_state and new_state (e.g. if an event occurred).
    */
     void                 re_adjust_impl(const T* new_vector);

    /**
     * @brief Validate initial conditions implementation.
     * @param t0 Initial time.
     * @param q0 Initial state.
     * @return True if ICs are valid.
     * @note Derived should call base first, then add additional checks.
     */
     bool                 validate_ics_impl(T t0, const T* q0) const;
    // ================================================================================


    // =========================== HELPER METHODS =====================================

    /// @brief Same as this->Rhs, but increments the RHS evaluation counter.
     void                 rhs(T* dq_dt, const T& t, const T* q) const;

    /// @brief Same as this->Jac, but increments the Jacobian evaluation counter.
     void                 jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const;

    /// @brief Get pointer to the initial conditions state data.
     const T*             ics_ptr() const;

    /// @brief Get pointer to the most recently computed state.
     const T*             new_state_ptr() const;

    /// @brief Get pointer to the previous accepted state.
     const T*             old_state_ptr() const;

    /// @brief Get pointer to the current "true" state.
     const T*             true_state_ptr() const;

    /// @brief Get pointer to the previous "true" state.
     const T*             last_true_state_ptr() const;

    /// @brief Get pointer to the correct new state for interpolation
     const T*             interp_new_state_ptr() const;

    /// @brief Get the time value of the newest computed step.
    const T&                    t_new() const;

    /// @brief Get the time value of the last "true" state.
    const T&                    t_last() const;

    /// @brief Print a warning that the solver is paused.
    void                        warn_paused() const;

    /// @brief Print a warning that the solver is dead.
    void                        warn_dead() const;

    /// @brief Set the solver status message.
     void                 set_message(const std::string& text);

    /**
    @brief Trigger re-adjustment right before state changes.
    @param new_vector New state vector values (size Nsys).

    @note The current state will be set to (t(), stepsize(), new_vector),
    where t() is the true current time, which might lie between old_state and new_state (e.g. if an event occurred). After that, any interpolation algorithms can only be valid in the interval [t_old, t), and the dense state vector output will reflect that with a discontinuity at t().
    */
    void                        re_adjust(const T* new_vector);

    /// @brief Check if the current true state matches the new state.
     bool                 is_at_new_state() const;

    // ================================================================================

    // ============================ OVERRIDEN IN RICH SOLVER ==========================
    /// @brief Time accessor implementation (overridden in RichSolver).
     const T&             t_impl() const;

    /// @brief Vector accessor implementation (overridden in RichSolver).
     const T*             vector_impl() const;

    /// @brief Advance implementation (overridden in RichSolver).
     bool                 adv_impl();
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

    void                    jac_exact(T* j, const T& t, const T* q) const;
    const T*                aux_state_ptr() const;
    T*                      aux_state_ptr();
    void                    register_states();
    bool                    validate_it(const T* state);
    void                    update_state(const T& time);
    void                    move_state(const T& time);
    void                    set_state(const T& time, T* state);


    Array2D<T, 6, (N>0 ? N+2 : 0), Allocation::Auto>    _state_data;
    mutable Array1D<T, 4*N, Allocation::Auto>           _dummy_state;
    Array1D<T, 4, Allocation::Stack>                    _scalar_data;
    Array1D<T>                                          _args;
    OdeData<RhsType, JacType>                           _ode;
    size_t                                              _Nsys = N;
    size_t                                              _Nupdates = 0;
    mutable size_t                                      _n_evals_rhs = 0;
    mutable size_t                                      _n_evals_jac = 0;
    std::string                                         _message = "Running";
    std::string                                         _name = name;
    int                                                 _direction = 1;
    int                                                 _new_state_idx = 1;
    int                                                 _old_state_idx = 2;
    int                                                 _true_state_idx = 1;
    int                                                 _last_true_state_idx = 2;
    int                                                 _aux_state_idx = 3;
    int                                                 _aux2_state_idx = 4;
    bool                                                _is_dead = false;
    bool                                                _diverges = false;
    bool                                                _is_running = true;
    bool                                                _use_new_state = true; //for interpolation purposes

    static constexpr int rtol_idx = 0;
    static constexpr int atol_idx = 1;
    static constexpr int min_step_idx = 2;
    static constexpr int max_step_idx = 3;
};


} // namespace ode

#endif
