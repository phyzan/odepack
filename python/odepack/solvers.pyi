import numpy as np
from typing import Iterable
from .events import *


class LowLevelFunction:
    """
    Wrapper for compiled C/C++ ODE right-hand side function.

    LowLevelFunction encapsulates a pointer to a C/C++ compiled function that
    computes the ODE right-hand side. These are created internally by the OdeSystem
    class when compiling symbolic ODE definitions.

    Users typically do not instantiate this directly. Instead, extract instances via
    OdeSystem.lowlevel_odefunc property.

    The wrapped function signature is: f(t, q, *args) -> output_array
    where *args are scalar float values.

    Notes
    -----
    This class bridges the gap between symbolic/Python ODE definitions and the
    high-performance C++ compiled implementations. It enables the same ODE to be
    solved with different numerical precision types.

    Parameters
    ----------
    pointer : int
        Internal C/C++ function pointer (opaque, set by OdeSystem).

    input_size : int
        Length of the state vector q.

    output_shape : tuple of int
        Expected output array shape. Usually matches input_size or is compatible.

    Nargs : int
        Number of extra scalar arguments the function accepts.

    scalar_type : str, optional
        Numerical type ("double", "float", "long double", "mpreal"). Default is "double".

    Examples
    --------
    Do not directly instantiate. Instead use OdeSystem:

    >>> from odepack import *
    >>> t, x, y = symbols('t, x, y')
    >>> ode_sys = OdeSystem(ode_sys=[y, -x], t=t, q=[x, y])
    >>> compiled_f = ode_sys.lowlevel_odefunc()  # Returns LowLevelFunction
    >>> q = np.array([1.0, 0.0])
    >>> dq_dt = compiled_f(0.0, q)
    """

    def __init__(self, pointer, input_size: int, output_shape: Iterable[int], Nargs: int, scalar_type: str = "double"):...

    def __call__(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        """
        Evaluate the compiled ODE function.

        Parameters
        ----------
        t : float
            Time parameter.

        q : np.ndarray
            State vector of length input_size.

        *args : float
            Additional scalar arguments (number must match Nargs).

        Returns
        -------
        np.ndarray
            Right-hand side values with shape output_shape.
        """
        ...

    @property
    def scalar_type(self)->str:
        """
        The numerical precision type used in the compiled function.

        Returns
        -------
        str
            One of "double", "float", "long double", "mpreal".
        """
        ...


class OdeSolver:
    """
    Base class for ODE solvers providing low-level step-by-step integration.

    OdeSolver is an iterator-like class that advances the solution of an ODE
    one adaptive step at a time. It provides access to intermediate solution values,
    solver diagnostics, and event handling. Use this class for fine-grained control
    over the integration process.

    This is an abstract base class. Use concrete implementations like RK23, RK45,
    DOP853, or BDF.

    See Also
    --------
    RK23 : Explicit Runge-Kutta 2(3) method
    RK45 : Explicit Runge-Kutta 4(5) method
    DOP853 : Explicit Runge-Kutta 8(5,3) method
    BDF : Implicit backward differentiation formula method
    RK4 : Fixed-step classic Runge-Kutta 4th order method
    """

    @property
    def t(self)->float:
        """
        The current integration time.

        Returns
        -------
        float
            Current value of the integration variable at the latest step.
        """
        ...

    @property
    def q(self)->np.ndarray:
        """
        The current state vector.

        Returns
        -------
        np.ndarray
            State vector q(t) at the current time, with the same shape as initial condition.
        """
        ...

    @property
    def t_old(self)->float:
        """
        The previous integration time that was automatically adapted using the solver's method.
        Events are not considered.

        Returns
        -------
        float
            Value of the integration variable at the step before the current one.
        """
        ...

    @property
    def q_old(self)->np.ndarray:
        """
        The previous state vector before the current step, corresponding to t_old.
        """
        ...

    @property
    def stepsize(self)->float:
        """
        The current adaptive step size.

        Returns
        -------
        float
            Step size that will be used for the next step.
        """
        ...

    @property
    def is_dead(self)->bool:
        """
        Check if the solver has reached a terminal state.

        Returns
        -------
        bool
            True if the solver cannot advance further, False otherwise.
        """
        ...

    @property
    def diverges(self)->bool:
        """
        Check if the solution has diverged.

        Returns
        -------
        bool
            True if integration stopped due to divergence (at least one state
            component became infinite or NaN), False otherwise. Implies is_dead.
        """
        ...

    @property
    def Nsys(self)->int:
        """
        The dimension of the ODE system.

        Returns
        -------
        int
            Number of equations in the system (length of state vector).
        """
        ...

    @property
    def n_evals_rhs(self)->int:
        """
        Get the number of RHS function evaluations performed so far.

        Returns
        -------
        int
            Total count of RHS evaluations.
        """
        ...

    @property
    def at_event(self)->bool:
        """
        Check if the solver is currently at any event time.

        Returns
        -------
        bool
            True if the current step landed exactly on an event time, False otherwise.
        """
        ...

    @property
    def status(self)->str:
        """
        The current status message of the solver.

        Returns
        -------
        str
            A descriptive message indicating the solver's current state.
        """
        ...

    def event_located(self, event: str)->bool:
        """
        Check if a specific event is located at the current step.

        Parameters
        ----------
        event : str
            Name of the event to check.

        Returns
        -------
        bool
            True if the specified event was detected at the current step, False otherwise.
        """
        ...

    def show_state(self, digits: int = 8)->None:
        """
        Print detailed information about the solver's current state.

        Displays current time, state vector, step size, and other diagnostics
        with the specified precision.

        Parameters
        ----------
        digits : int, optional
            Number of significant digits for floating-point display. Default is 8.
        """

    def advance(self)->bool:
        """
        Advance the solver by a single adaptive time step.

        Computes the next step of the integration and updates the internal state.
        For event-free problems, this is the basic iteration method.

        Returns
        -------
        bool
            True if the solver successfully advanced. False if the solver could not
            advance further (e.g., divergence detected, validation failure, or solver
            stopped by an event/user). When False, a diagnostic message is printed
            explaining why advancement failed.
        """

    def rhs(self, t: float, q: np.ndarray)->np.ndarray:
        """
        Evaluate the right-hand side function at a given time and state.

        Parameters
        ----------
        t : float
            Time at which to evaluate the RHS.

        q : np.ndarray
            State vector at which to evaluate the RHS.

        Returns
        -------
        np.ndarray
            The computed right-hand side values f(t, q).
        """
        ...

    def jac(self, t: float, q: np.ndarray)->np.ndarray:
        """
        Evaluate the Jacobian matrix of the RHS function at a given time and state.

        Parameters
        ----------
        t : float
            Time at which to evaluate the Jacobian.

        q : np.ndarray
            State vector at which to evaluate the Jacobian.

        Returns
        -------
        np.ndarray
            The computed Jacobian matrix J(t, q) where J[i, j] = df_i/dq_j.
            If the solver was not instanciated with an analytical Jacobian, this method returns a finite difference approximation.
        """
        ...

    def advance_to_event(self)->bool:
        """
        Advance the solver until the next event occurrence.

        Continuously steps the solver forward by individual steps until an event is
        detected. If no more events are available, advances to the end of the
        integration interval.

        Returns
        -------
        bool
            True if an event was detected and the solver advanced to it. False if
            no more events are available or advancement failed. When False, a
            diagnostic message is printed explaining the reason.

        Note
        ----
        This requires events to be registered in the solver at initialization.
        """

    def advance_until(self, t: float)->bool:
        """
        Advance the solver until precisely reaching a specified time.
        If the target time t falls between steps, the solver will perform
        a specialized interpolation step to land exactly on t, so that
        self.t == t after this call, and self.q is the solution at time t.

        Continuously steps the solver forward by individual steps until the
        integration time t is reached.

        Parameters
        ----------
        t : float
            Target time to advance to.

        Returns
        -------
        bool
            True if the solver successfully advanced to t.
            False if advancement failed before reaching t. When False, a diagnostic
                message is printed explaining the reason. Alternatively, check self.status.
        """

    def reset(self)->None:
        """
        Reset the solver to its initial conditions.

        Restores the solver to (t0, q0) with all internal state reset.
        The solver is ready to begin integration again from the start.
        """

    def resume(self)->bool:
        """
        Resume the solver after being stopped.

        If the solver was previously halted due to an event or user stop,
        this method allows continuation of integration from the current state.

        Returns
        -------
        bool
            True if the solver was successfully resumed.
            False if the solver is not in a stoppable state (e.g., dead).
            If False, call message to see the reason.
        """

    def set_ics(self, t0: float, q0: np.ndarray, stepsize: float = 0., direction: int = 0)->bool:
        """
        Set new initial conditions for the solver.

        This method allows re-initializing the solver with a new starting time
        and state vector. Optionally, an initial step size can be provided.

        Parameters
        ----------
        t0 : float
            New initial time.

        q0 : np.ndarray
            New initial state vector.

        stepsize : float, optional
            Initial step size estimate. If 0 (default), automatically determined.

        direction : int, optional
            Integration direction. 1 for forward, -1 for backward, 0 for unchanged.
            Default is 0 (keep current direction).

        Returns
        -------
        bool
            True if the initial conditions were successfully set.
            False if there was an error (e.g., nan/inf values).
        """

    def copy(self)->OdeSolver:
        """
        Create a deep copy of the solver in its current state.

        Returns
        -------
        OdeSolver
            A new solver object with identical internal state, history, and configuration.
            Modifying the copy does not affect the original.
        """

    @property
    def scalar_type(self)->str:
        """
        The numerical precision type used for all computations.

        Returns
        -------
        str
            One of "double", "float", "long double", or "mpreal".
        """


class RK23(OdeSolver):
    """
    Explicit Runge-Kutta method of order 2(3).

    A low-order, low-computational-cost explicit Runge-Kutta method suitable for
    non-stiff problems with modest accuracy requirements. Uses an adaptive step size
    based on embedded Runge-Kutta formulas of order 2 and 3.

    This solver is faster but less accurate than RK45. Choose RK23 for:
    - Problems requiring lower accuracy
    - Non-stiff systems
    - When computational speed is more important than accuracy

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    rtol, atol : float, optional
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    stepsize : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".

    Examples
    --------
    >>> def f(t, q):
    ...     return np.array([q[1], -q[0]])
    >>> solver = RK23(f, 0, np.array([1.0, 0.0]), rtol=1e-6, atol=1e-9)
    >>> # Advance 10 steps
    >>> for _ in range(10):
    ...     if solver.advance():
    ...         print(f"t={solver.t:.3f}, q={solver.q}")
    """

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        ...


class RK45(OdeSolver):
    """
    Explicit Runge-Kutta method of order 4(5) (Dormand-Prince RK45).

    A general-purpose explicit Runge-Kutta method suitable for most non-stiff problems.
    Uses an adaptive step size based on embedded formulas of order 4 and 5. Offers
    a good balance between accuracy and computational cost.

    This is the recommended solver for non-stiff problems. Choose RK45 for:
    - General-purpose non-stiff ODE integration
    - Moderate to high accuracy requirements
    - Most smooth problems

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    rtol, atol : float, optional
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    stepsize : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".

    Examples
    --------
    >>> def f(t, q):
    ...     return np.array([q[1], -q[0]])
    >>> solver = RK45(f, 0, np.array([1.0, 0.0]), rtol=1e-6, atol=1e-9)
    >>> # Integrate until t=1.0
    >>> while solver.t < 1.0 and not solver.is_dead:
    ...     solver.advance()
    """

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        ...


class DOP853(OdeSolver):
    """
    Explicit Runge-Kutta method of order 8(5,3) (Dormand-Prince DOP853).

    A high-order explicit Runge-Kutta method for non-stiff problems requiring high
    accuracy. Uses embedded formulas of order 8, 5, and 3 for sophisticated error
    control. Computationally expensive but provides excellent accuracy for smooth problems.

    Choose DOP853 for:
    - Very high accuracy requirements
    - Smooth non-stiff problems
    - When computational cost is not the primary concern

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    rtol, atol : float, optional
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    stepsize : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".

    Examples
    --------
    >>> def f(t, q):
    ...     return np.array([q[1], -q[0]])
    >>> solver = DOP853(f, 0, np.array([1.0, 0.0]), rtol=1e-13, atol=1e-13)
    >>> # Integrate to t=10
    >>> while solver.t < 10.0 and not solver.is_dead:
    ...     solver.advance()
    """

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        ...


class BDF(OdeSolver):
    """
    Implicit backward differentiation formula (BDF) method.

    A variable-order implicit method designed for stiff ODEs. Uses the Jacobian matrix
    to solve implicit equations at each step. Much more computationally expensive than
    explicit methods but handles stiff problems that would require prohibitively small
    step sizes with explicit methods.

    Choose BDF for:
    - Stiff ODE systems
    - Problems with widely separated timescales
    - When explicit methods fail or become too slow

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    jac : callable, optional
        Jacobian of f: jac(t, q, *args) -> matrix.
        Must return a matrix of shape (len(q), len(q)) where jac[i, j] = df_i/dq_j.
        If None, central finite differences will be used to approximate the Jacobian,
            using adaptive step sizes.

    rtol, atol : float, optional
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    stepsize : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f, jac, and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".

    Notes
    -----
    The Jacobian function jac is required. Providing an accurate Jacobian is important
    for performance. If unavailable, consider using an explicit method on a transformed
    or reduced version of the problem.

    Examples
    --------
    >>> def f(t, q):
    ...     # Stiff system
    ...     return np.array([-1000*q[0], q[0] - q[1]])
    >>> def jac(t, q):
    ...     return np.array([[-1000, 0], [1, -1]])
    >>> solver = BDF(f, 0, np.array([1.0, 0.0]), jac=jac, rtol=1e-6, atol=1e-9)
    >>> # Integrate to t=1.0
    >>> while solver.t < 1.0 and not solver.is_dead:
    ...     solver.advance()
    """

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, jac: Func = None, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        ...


class RK4(OdeSolver):
    """
    Explicit Runge-Kutta method of order 4(5). This is the standard Runge-Kutta 4 method.

    The core algorithm consists of 4 stages:

    k1 = f(t, q)
    k2 = f(t + h/2, q + h/2 * k1)
    k3 = f(t + h/2, q + h/2 * k2)
    k4 = f(t + h, q + h * k3)

    And the state is updated as:
    q_new = q + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    while keeping the stepsize fixed.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    rtol, atol : float, optional
        Not used to adapt the stepsize, but is used to estimate the first step size, if that is set to 0.

    min_step, max_step : They are not used, as the stepsize is fixed. They are only present for API consistency.

    stepsize : float, optional
        The fixed step size to use for integration. Default is 0., which means a step size is estimated initially based on rtol and atol.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".
    """

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        ...


def advance_all(solvers: Iterable[OdeSolver], t_goal: float, threads=-1, display_progress = False)->None:
    """
    Advance multiple OdeSolver objects to a target time in parallel.

    This function provides efficient batch advancement of low-level OdeSolver objects
    using multi-threaded parallelization. Unlike integrate_all which operates on
    LowLevelODE objects (which accumulate integration history), advance_all operates
    on the underlying OdeSolver objects which maintain only their current state.

    Each OdeSolver is advanced step-by-step from its current time to the target time
    t_goal, updating only its internal state (t, q, t_old, q_old) at each step without
    accumulating history. This is useful for advancing solvers obtained via the
    LowLevelODE.solver() method or when working directly with step-by-step integration.

    All solvers must use compiled functions (compiled=True). Solvers with different
    scalar types (double, float, long double, mpreal) are automatically grouped for
    efficient processing.

    Parameters
    ----------
    solvers : iterable of OdeSolver
        Collection of OdeSolver objects to advance. Each solver will be advanced from
        its current time to t_goal. All solvers must use compiled functions (not pure
        Python functions). Can include any OdeSolver subclass, as long as they inherit from
        the provided OdeSolver subclasses (e.g. RK45, BDF, etc.), and they have been constructed
        using fully compiled functions. If a custom OdeSolver subclass is passed that overrides
        the advance() method, that will NOT be called while integrating the solver to the requested
        t_goal. Instead, the compiled advance() method is used internally.

    t_goal : float
        Target time. Each solver advances from its current time (solver.t) to this
        target time using its adaptive stepping algorithm.

    threads : int, optional
        Number of worker threads for parallel execution.
        - 0 or -1 (default): Use number of CPU threads
        - n >= 1: Use n threads

    display_progress : bool, optional
        If True, print progress information during advancement. Default is False.

    Raises
    ------
    ValueError
        If any solver uses pure Python functions (compiled=False).
        If a list item is not a recognized OdeSolver object type.

    Notes
    -----
    All solvers are advanced synchronously. The function returns when all have
    reached t_goal.

    Unlike integrate_all (which works with LowLevelODE objects that accumulate
    integration history), advance_all works with the underlying OdeSolver objects
    that only maintain current state (t, q, t_old, q_old). The solvers are modified
    in place.

    Examples
    --------
    Advance multiple solvers to the same target time:

    >>> from odepack import *
    >>> t, x, v = symbols('t, x, v')
    >>> system = OdeSystem(ode_sys=[v, -x], t=t, q=[x, v])
    >>> # Create LowLevelODE objects
    >>> odes = [system.get(t0=0, q0=[1.0+0.01*i, 0.0], compiled=True) for i in range(10)]
    >>> # Extract underlying solvers
    >>> solvers = [ode.solver() for ode in odes]
    >>> # Advance all solvers to t=10.0
    >>> advance_all(solvers, t_goal=10.0, threads=4)
    >>> # Check final states
    >>> print(f"First solver at t={solvers[0].t:.3f}, q={solvers[0].q}")
    """
    ...


def set_mpreal_prec(bits: int)->None:
    """
    Set the global precision for MPFR arbitrary-precision arithmetic.

    Sets the bit precision used by the "mpreal" scalar type (arbitrary precision
    floating-point numbers via MPFR library). This precision applies to all new
    mpreal objects created after this call.

    The default precision is 53 bits, which is equivalent to IEEE 754 double
    precision.

    Parameters
    ----------
    bits : int
        Number of bits of precision. Examples:
        - 53 (default): Equivalent to double precision (Python's float)
        - 128: ~38 decimal digits
        - 256: ~77 decimal digits
        - 1024: ~308 decimal digits

    Notes
    -----
    Higher precision costs more memory and computation time. Set only as high as
    needed for your application.

    This is a global setting affecting the entire process. All new mpreal objects
    created after calling this function will use the specified precision.

    Examples
    --------
    >>> set_mpreal_prec(128)  # Set to 128-bit precision
    >>> ode = LowLevelODE(f, 0, q0, scalar_type="mpreal")
    >>> # Subsequent ODEs will use 128-bit precision
    """
    ...

def mpreal_prec()->int:
    """
    Get the current global MPFR precision in bits.

    Returns
    -------
    int
        Current bit precision for mpreal scalar type.

    Examples
    --------
    >>> precision = mpreal_prec()
    >>> print(f"Current precision: {precision} bits")
    """
    ...
