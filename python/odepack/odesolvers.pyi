from __future__ import annotations
import numpy as np
from typing import Callable, Iterable, overload, Any

Func = Callable[[float, np.ndarray], np.ndarray]
ObjFunc = Callable[[float, np.ndarray], float]
BoolFunc = Callable[[float, np.ndarray], bool]


class EventOpt:
    """
    Configuration options for controlling event behavior during ODE integration.

    This class allows fine-tuned control over how events are detected, recorded, and
    handled during ODE integration. Use this class to specify per-event behavior such
    as maximum number of events to store, integration termination on event, and
    periodic event filtering.

    Parameters
    ----------
    name : str
        Name of the event these options apply to. Must match the name provided in
        the corresponding Event object passed to the ODE solver.

    max_events : int, optional
        Maximum number of events to store and process before stopping to count events.
        - If max_events = -1 (default): All events are stored.
        - If max_events = 0: All events are ignored.
        - If max_events > 0: Integration stores at most this many events.

    terminate : bool, optional
        If True, halt ODE integration once max_events have been encountered.
        Only effective when max_events > 0. Default is False.

    period : int, optional
        Store events at periodic intervals. Only every period-th event is stored.
        - If period = 1 (default): All events are stored.
        - If period = 2: First event skipped, second stored, third skipped, etc.
        - If period = n: Store every n-th encountered event.

    Examples
    --------
    >>> # Store all events
    >>> opt1 = EventOpt("my_event", max_events=-1)
    >>> # Stop integration after 10 events
    >>> opt2 = EventOpt("my_event", max_events=10, terminate=True)
    >>> # Store every other event
    >>> opt3 = EventOpt("my_event", max_events=-1, period=2)
    """

    def __init__(self, name: str, max_events = -1, terminate = False, period=1):...


class Event:
    """
    Base class for events that can be detected during ODE integration.

    This is an abstract base class. Use subclasses like PreciseEvent or PeriodicEvent
    to define actual events. An event is a condition that is detected and accurately
    located during ODE integration, potentially with state vector modification.

    Properties
    ----------
    name : str
        The name identifier of the event.

    hides_mask : bool
        If True, the event's mask function is hidden from the result at the event time.

    scalar_type : str
        The numerical precision type used for event computations.
    """

    @property
    def name(self)->str:
        """
        The name identifier of the event.

        Returns
        -------
        str
            Event name.
        """
        ...

    @property
    def hides_mask(self)->bool:
        """
        Whether the mask function result is hidden in the solution at event time.

        Returns
        -------
        bool
            True if mask is hidden, False otherwise.
        """
        ...

    @property
    def scalar_type(self)->str:
        """
        The numerical scalar type used for event computations.

        Returns
        -------
        str
            One of "double", "float", "long double", or "mpreal".
        """
        ...


class PreciseEvent(Event):
    """
    An event detected when an objective function crosses zero with precise timing.

    This event type detects when a continuous objective function changes sign during
    ODE integration and accurately determines the exact time of the zero crossing using
    a root solver. The event can optionally trigger a state vector modification (mask)
    at the event time.

    Parameters
    ----------
    name : str
        Unique identifier for this event.

    when : callable
        Objective function with signature when(t, q, *args) -> float.
        The event is detected when this function's sign changes. The solver will
        accurately determine the time t where when(t, q(t)) = 0.

    direction : {-1, 0, 1}, optional
        Controls which type of zero crossing is detected:
        - 0 (default): Any direction (rising or falling).
        - 1: Only rising zero crossings (when(t-) < 0 and when(t+) > 0).
        - -1: Only falling zero crossings (when(t-) > 0 and when(t+) < 0).

    mask : callable, optional
        State modification function with signature mask(t, q, *args) -> array.
        If provided, the state vector q is updated as q := mask(t, q) at the event time.
        The returned array must have the same shape as q.

    hide_mask : bool, optional
        If True, the mask is not applied to the OdeResult at the event time. The solution
        will appear as if the mask was not applied, while the solver internally uses the
        masked value for continuation. Default is False.

    event_tol : float, optional
        Tolerance for the root solver that accurately determines event time.
        Default is 1e-12. Smaller values give more accurate event timing but cost more.

    scalar_type : str, optional
        Numerical precision type. One of:
        - "double" (default)
        - "float"
        - "long double"
        - "mpreal" (arbitrary precision via MPFR)

    __Nsys : int, optional
        Internal parameter. Only set by OdeSystem when instantiating events from
        compiled functions (when using symbolic ODE definitions with low-level
        function pointers). Do not use directly.

    __Nargs : int, optional
        Internal parameter. Only set by OdeSystem when instantiating events from
        compiled functions. Do not use directly.

    Examples
    --------
    Detect when x-component reaches 1 during a 2D oscillation:

    >>> def f(t, q):
    ...     return np.array([q[1], -q[0]])
    >>> def event_condition(t, q):
    ...     return q[0] - 1  # Triggers when x = 1
    >>> event = PreciseEvent("x_equals_1", event_condition, event_tol=1e-12)

    Detect event with direction and mask:

    >>> def event_condition(t, q):
    ...     return q[0] - 1
    >>> def mask(t, q):
    ...     q_new = q.copy()
    ...     q_new[1] *= -1  # Reverse velocity component
    ...     return q_new
    >>> event = PreciseEvent(
    ...     "bouncing_wall", event_condition, direction=1, mask=mask
    ... )
    """

    def __init__(self, name: str, when: ObjFunc, direction=0, mask: Func=None, hide_mask=False, event_tol=1e-12, scalar_type: str = "double", __Nsys: int = 0, __Nargs: int = 0):
        pass

    @property
    def event_tol(self)->float:
        """
        The tolerance used for accurately determining event time.

        Returns
        -------
        float
            Root solver tolerance for event time determination.
        """
        ...


class PeriodicEvent(Event):
    """
    An event triggered periodically at fixed time intervals.

    This event type triggers at regular time intervals specified by period.
    Unlike PreciseEvent, the timing is pre-determined by the
    period rather than detected from a function value.

    Parameters
    ----------
    name : str
        Unique identifier for this event.

    period : float
        Time interval between successive events. Must be positive.

    mask : callable, optional
        State modification function with signature mask(t, q, *args) -> array.
        If provided, the state vector q is updated at each event time.

    hide_mask : bool, optional
        If True, the mask is not shown in the solution at event times.
        Default is False.

    scalar_type : str, optional
        Numerical precision type. One of:
        - "double" (default)
        - "float"
        - "long double"
        - "mpreal"

    __Nsys : int, optional
        Internal parameter. Do not use.

    __Nargs : int, optional
        Internal parameter. Do not use.

    Examples
    --------
    Record solution every 0.1 time units:

    >>> event = PeriodicEvent("record", period=0.1)

    Apply periodic velocity reversal (e.g., for perturbation):

    >>> def kick(t, q):
    ...     q_new = q.copy()
    ...     q_new[1] *= -1  # Reverse velocity
    ...     return q_new
    >>> event = PeriodicEvent("kick", period=2.0, mask=kick)
    """

    def __init__(self, name: str, period: float, mask: Func=None, hide_mask=False, scalar_type: str = "double", __Nsys: int = 0, __Nargs: int = 0):...

    @property
    def period(self)->float:
        """
        The time interval between successive events.

        Returns
        -------
        float
            Period in time units.
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
        pass

    @property
    def q(self)->np.ndarray:
        """
        The current state vector.

        Returns
        -------
        np.ndarray
            State vector q(t) at the current time, with the same shape as initial condition.
        """
        pass

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
        pass

    @property
    def q_old(self)->np.ndarray:
        """
        The previous state vector before the current step, corresponding to t_old.
        """
        pass

    @property
    def stepsize(self)->float:
        """
        The current adaptive step size.

        Returns
        -------
        float
            Step size that will be used for the next step.
        """
        pass

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

    def set_ics(self, t0: float, q0: np.ndarray, dt: float = 0.)->bool:
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

        dt : float, optional
            Initial step size estimate. If 0 (default), automatically determined.

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

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
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

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        pass


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

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
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

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        pass


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

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
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

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        pass


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

    jac : callable
        Jacobian of f: jac(t, q, *args) -> matrix.
        Must return a matrix of shape (len(q), len(q)) where jac[i, j] = df_i/dq_j.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
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
    >>> solver = BDF(f, jac, 0, np.array([1.0, 0.0]), rtol=1e-6, atol=1e-9)
    >>> # Integrate to t=1.0
    >>> while solver.t < 1.0 and not solver.is_dead:
    ...     solver.advance()
    """

    def __init__(self, f: Func, jac: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        pass


class VariationalSolver(OdeSolver):
    """
    Low-level ODE solver for variational equations with real-time Lyapunov exponent tracking.

    VariationalSolver is the step-by-step iterator for variational equations, similar to
    how OdeSolver iterates through standard ODE solutions. Unlike VariationalLowLevelODE
    (which accumulates full integration history), VariationalSolver maintains only the
    current state and provides real-time access to Lyapunov exponent calculations.

    This class integrates both the primary system and its variational equations
    (linearized perturbation dynamics) simultaneously. At regular intervals (determined
    by the period parameter), the variational state is renormalized to prevent numerical
    overflow, and Lyapunov exponent metrics are updated.

    The state vector has even length: the first half is the primary state, the second
    half is the variational state (perturbation vector). The variational state is
    automatically normalized to unit length at initialization and at each renormalization.

    Parameters
    ----------
    f : callable
        Right-hand side function for the augmented variational system: f(t, q, *args) -> array.
        The input q has even length (primary state + variational state), and the output
        must match this length. Typically obtained from compiled variational equations.

    jac : callable
        Jacobian matrix function for the augmented system: jac(t, q, *args) -> matrix.
        Required for BDF method. The Jacobian should have shape (2*Nsys, 2*Nsys) for
        the augmented variational system.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector with even length. The first half is the primary state,
        the second half is the initial perturbation direction. The perturbation is
        automatically normalized to unit length at initialization.

    period : float
        Renormalization period for the variational state. The perturbation vector is
        renormalized to unit length every 'period' time units. The growth rate at each
        renormalization contributes to the Lyapunov exponent calculation.

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward.

    args : tuple, optional
        Extra arguments passed to f and jac. Default is ().

    method : str, optional
        Integration method: "RK23", "RK45" (default), "DOP853", or "BDF".

    scalar_type : str, optional
        Numerical precision: "double" (default), "float", "long double", or "mpreal".

    Attributes
    ----------
    t : float
        Current integration time (inherited from OdeSolver).

    q : np.ndarray
        Current state vector (primary state + variational state) (inherited from OdeSolver).

    logksi : float
        Cumulative logarithm of the variational state norm growth.
        This is the sum of log(|delta_q|) at each renormalization event.

    lyap : float
        Current estimate of the Lyapunov exponent: logksi / t_lyap.
        This is the average exponential growth rate of the perturbation.

    t_lyap : float
        Total time elapsed for Lyapunov exponent computation.
        This is the time since the first renormalization event.

    delta_s : float
        Most recent logarithmic growth "kick" from the last renormalization:
        delta_s = log(|delta_q|) before normalization.

    Notes
    -----
    This is a low-level iterator class. For most use cases, prefer using
    OdeSystem.get_variational() which returns VariationalLowLevelODE (a higher-level
    wrapper that accumulates history).

    The Lyapunov exponent is computed as:
    lambda = logksi / t_lyap = (1/T) * sum(log(|delta_q_i|))

    Positive Lyapunov exponents indicate chaos, negative indicate stability, and
    zero indicates neutral directions.

    Examples
    --------
    Create a variational solver for a simple oscillator:

    >>> from odepack import *
    >>> t, x, v = symbols('t, x, v')
    >>> system = OdeSystem(ode_sys=[v, -x], t=t, q=[x, v])
    >>> # Get compiled variational functions
    >>> f_ptr, jac_ptr = system._pointers(scalar_type='double', variational=True)[:2]
    >>> # Initial state: [x0, v0, dx, dv]
    >>> q0 = np.array([1.0, 0.0, 1.0, 0.0])
    >>> solver = VariationalSolver(
    ...     f=f_ptr, jac=jac_ptr, t0=0, q0=q0, period=1.0,
    ...     method="RK45", scalar_type="double"
    ... )
    >>> # Step through integration
    >>> while solver.t < 10.0:
    ...     solver.advance()
    >>> print(f"Lyapunov exponent: {solver.lyap}")

    See Also
    --------
    VariationalLowLevelODE : High-level wrapper with history accumulation
    OdeSystem.get_variational : Recommended way to create variational solvers
    OdeSystem.get_var_solver : Alternative method returning VariationalSolver directly
    OdeSolver : Base class for step-by-step ODE integration
    """

    def __init__(self, f: Func, jac: Func, t0: float, q0: np.ndarray, period: float, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), method: str = "RK45", scalar_type: str = "double"):
        ...

    @property
    def logksi(self)->float:
        """
        Cumulative logarithm of variational state norm growth.

        Returns
        -------
        float
            Sum of log(|delta_q|) over all renormalization events, where delta_q is
            the variational state vector before normalization. This accumulates the
            total logarithmic growth of the perturbation.
        """
        ...

    @property
    def lyap(self)->float:
        """
        Current Lyapunov exponent estimate.

        Returns
        -------
        float
            The Lyapunov exponent: logksi / t_lyap. This is the average exponential
            growth rate of perturbations. Positive values indicate chaos, negative
            values indicate stability, and values near zero indicate neutral dynamics.
        """
        ...

    @property
    def t_lyap(self)->float:
        """
        Total time elapsed for Lyapunov exponent computation.

        Returns
        -------
        float
            Time since the first renormalization event. Used as the denominator in
            the Lyapunov exponent calculation: lyap = logksi / t_lyap.
        """
        ...

    @property
    def delta_s(self)->float:
        """
        Most recent logarithmic growth from the last renormalization.

        Returns
        -------
        float
            The logarithm of the variational state norm at the most recent
            renormalization: log(|delta_q|). This represents the "kick" or
            instantaneous growth contribution from the last renormalization period.
        """
        ...


class LowLevelODE:
    """
    Container for an ODE problem with dynamic solution history accumulation.

    LowLevelODE is a high-level wrapper around OdeSolver that accumulates and manages
    the complete integration history. Unlike OdeSolver (which focuses on step-by-step
    iteration), LowLevelODE provides convenient methods to integrate over intervals
    and automatically stores all results.

    The object grows dynamically as integration methods are called. Each call to
    integrate(), go_to(), or rich_integrate() appends new results to the internal
    history.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE: f(t, q, *args) -> array.
        Must return an array of the same shape as q0.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state vector.

    jac : callable, optional
        Jacobian matrix function: jac(t, q, *args) -> matrix.
        Required for BDF method. Matrix elements should be jac[i, j] = df_i/dq_j.

    rtol : float, optional
        Relative tolerance for adaptive step control. Default is 1e-12.

    atol : float, optional
        Absolute tolerance for adaptive step control. Default is 1e-12.
        Error estimate: atol + rtol * |q|

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    first_step : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward in time.

    args : tuple, optional
        Extra arguments passed to f, jac, and events. Default is ().

    events : iterable, optional
        Sequence of Event objects to detect. Default is ().

    method : str, optional
        Integration method. One of:
        - "RK23" (fastest, lowest accuracy)
        - "RK45" (recommended default)
        - "DOP853" (highest accuracy)
        - "BDF" (for stiff problems, requires jac)
        Default is "RK45".

    scalar_type : str, optional
        Numerical precision. One of "double" (default), "float", "long double", "mpreal".

    Examples
    --------
    Basic integration:

    >>> def f(t, q):
    ...     return np.array([q[1], -q[0]])
    >>> ode = LowLevelODE(f, 0, np.array([1.0, 0.0]))
    >>> result = ode.integrate(10.0)
    >>> print(result.t)

    Integration with events:

    >>> event = PreciseEvent("crossing", lambda t, q: q[0] - 1)
    >>> ode = LowLevelODE(f, 0, np.array([1.0, 0.0]), events=[event])
    >>> result = ode.integrate(10.0)
    >>> print(ode.event_map)
    """

    @overload
    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=None, first_step=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):
        pass

    def solver(self)->OdeSolver:
        """
        Get a copy of the current underlying OdeSolver.

        Returns
        -------
        OdeSolver
            A copy of the internal solver in its current state. Modifications to the
            returned solver do not affect this LowLevelODE object.
        """
        pass

    def integrate(self, interval, *, t_eval : Iterable[float] = None, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeResult:
        """
        Integrate the ODE over a time interval.

        Advances the solution forward by the specified interval and accumulates results.
        Results are added to the internal history. Calling integrate() multiple times
        continues from the previous endpoint.

        Parameters
        ----------
        interval : float
            Duration of integration. Must be positive (for forward integration).
            For backward integration (direction=-1), interval should still be positive;
            the direction is handled automatically.

        t_eval : array-like, optional
            Times at which to explicitly store solution values. If None (default),
            all steps encountered by the adaptive solver are stored. Values should be
            within [t_current, t_current + interval].

        event_options : iterable, optional
            Sequence of EventOpt objects controlling event behavior. One per event
            registered in this ODE. Default is ().

        max_prints : int, optional
            Progress display granularity. If 0 (default), no progress is printed.
            - max_prints = 100: Print at 1%, 2%, ..., 100%
            - max_prints = 1000: Print at 0.1%, 0.2%, ..., 100%
            Larger values show finer progress but may impact performance.

        Returns
        -------
        OdeResult
            Result containing the newly computed solution segment. The internal
            history of this LowLevelODE is updated with these results.
        """
        pass

    def go_to(self, t, *, t_eval : Iterable[float] = None, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeResult:
        """
        Integrate the ODE to a specific time target.

        Similar to integrate(), but the target is an absolute time rather than a duration.

        Parameters
        ----------
        t : float
            Target time to reach.

        t_eval : array-like, optional
            Times at which to store solution values. Must be within [t_current, t].

        event_options : iterable, optional
            Event configuration. Default is ().

        max_prints : int, optional
            Progress display granularity. Default is 0 (no output).

        Returns
        -------
        OdeResult
            Result containing the newly computed segment.
        """
        ...

    def rich_integrate(self, interval, *, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeSolution:
        """
        Integrate and return a solution object with interpolation capability.

        Like integrate(), but ensures all solver steps are retained to enable accurate
        continuous interpolation. Returns an OdeSolution object which can be called
        as a function to evaluate the solution at any time within the integration interval.

        This is more memory-intensive than integrate() but provides smooth interpolation.

        Parameters
        ----------
        interval : float
            Duration of integration.

        event_options : iterable, optional
            Event configuration. Default is ().

        max_prints : int, optional
            Progress display granularity. Default is 0.

        Returns
        -------
        OdeSolution
            Callable result object supporting interpolation via __call__(t).
            Solution values can be evaluated at any time in the integrated interval.
        """
        pass

    @property
    def t(self)->np.ndarray:
        """
        All recorded time steps in the accumulated history.

        Returns
        -------
        np.ndarray
            Sorted array of all times at which solutions have been stored.
        """
        ...

    @property
    def q(self)->np.ndarray:
        """
        All recorded solution state vectors in the accumulated history.

        Returns
        -------
        np.ndarray
            Array of shape (n_steps, n_states) where q[i] is the state at t[i].
        """
        ...

    @property
    def Nsys(self)->int:
        """
        The dimension of the ODE system.

        Returns
        -------
        int
            Number of equations (state vector length).
        """
        ...

    @property
    def runtime(self)->float:
        """
        Total computational time spent on integration.

        Returns
        -------
        float
            Wall-clock time in seconds for all integrate/go_to/rich_integrate calls.
        """
        ...

    @property
    def diverges(self)->bool:
        """
        Check if any integration has diverged.

        Returns
        -------
        bool
            True if divergence was detected during any integration step.
        """
        ...

    @property
    def is_dead(self)->bool:
        """
        Check if the underlying solver is in a terminal state.

        Returns
        -------
        bool
            True if the solver cannot advance further.
        """
        ...

    @property
    def event_map(self)->dict[str, np.ndarray[int]]:
        """
        Indices of solution steps where each event occurred.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping event names to arrays of step indices where events
            occurred. Use these indices to index into self.t and self.q.

        Examples
        --------
        >>> ode = LowLevelODE(f, 0, q0, events=[event1, event2])
        >>> ode.integrate(10.0)
        >>> idx_event1 = ode.event_map['event1']
        >>> t_at_event1 = ode.t[idx_event1]
        >>> q_at_event1 = ode.q[idx_event1]
        """
        ...

    @property
    def scalar_type(self)->str:
        """
        The numerical precision type used.

        Returns
        -------
        str
            One of "double", "float", "long double", "mpreal".
        """
        ...

    def copy(self)->LowLevelODE:
        """
        Create a deep copy of this ODE object.

        Returns
        -------
        LowLevelODE
            A new object with identical configuration and accumulated history.
        """
        ...

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:
        """
        Get times and states at which a specific event occurred.

        Convenience method combining event_map lookup with solution access.

        Parameters
        ----------
        event : str
            Name of the event to query.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair (t_event, q_event) where:
            - t_event: Array of times when event occurred
            - q_event: Array of states at those times

        Examples
        --------
        >>> t_evt, q_evt = ode.event_data('wall_collision')
        """
        ...

    def reset(self)->None:
        """
        Reset the solver to its initial conditions.

        Clears all history and returns the solver to (t0, q0).
        After reset, the internal history is empty except for the initial state.
        """
        ...

    def clear(self)->None:
        """
        Clear accumulated history while preserving current state.

        Empties the internal history but keeps the solver at its current (t, q).
        After clear:
        - self.t contains only [t_current]
        - self.q contains only [q_current]
        - Further integration continues from this state

        Use this to reduce memory consumption when intermediate solution values
        are no longer needed.
        """
        ...




class VariationalLowLevelODE(LowLevelODE):
    """
    ODE container for variational equations with Lyapunov exponent tracking.

    This specialized class extends LowLevelODE to automatically compute Lyapunov exponents
    during integration. It tracks how perturbations to the state vector grow or shrink
    along the solution trajectory.

    The state vector q0 must have an even number of elements: the first half represents
    the primary state, and the second half represents the variational state (perturbation).
    The variational state evolves according to the linearized equation:
    d(delta_q)/dt = Jacobian(f) * delta_q

    Parameters
    ----------
    f : callable
        Right-hand side: f(t, q, *args) -> array. q has even length.

    t0 : float
        Initial time.

    q0 : np.ndarray
        Initial state with even length. First half is primary state, second half
        is initial perturbation vector. The variational state (second half) is
        automatically normalized to unit length at initialization.

    period : float
        Renormalization period for the variational state. The perturbation is
        renormalized at regular intervals to prevent numerical overflow/underflow.

    jac : callable, optional
        Jacobian (required for BDF). jac(t, q, *args) -> matrix.
        The Jacobian should have shape (n_primary, n_primary) and be computed
        based on the primary state portion.

    rtol, atol : float, optional
        Relative and absolute tolerances. Default is 1e-12.

    min_step, max_step, first_step : float, optional
        Step size control. See LowLevelODE.

    direction : {-1, 1}, optional
        Integration direction. Default is 1 (forward).

    args : tuple, optional
        Extra arguments for f and jac. Default is ().

    events : iterable, optional
        Events to detect. Default is ().

    method : str, optional
        Integration method. Default is "RK45". For stiff systems, use "BDF".

    scalar_type : str, optional
        Numerical precision. Default is "double".

    Notes
    -----
    Lyapunov exponents measure the average exponential growth rate of small perturbations.
    Positive exponents indicate chaos, negative exponents indicate stability, and zero
    indicates neutral directions.

    The Lyapunov exponent is computed as:
    lambda = (1/T) * sum(log(|delta_q_i|)) over renormalization events

    Examples
    --------
    Compute Lyapunov exponent for a simple harmonic oscillator:

    >>> # Use OdeSystem.get_variational() to create variational systems
    >>> from odepack import *
    >>> t, x, v = symbols('t, x, v')
    >>> system = OdeSystem(ode_sys=[v, -x], t=t, q=[x, v])
    >>> # Initial conditions: [x0, v0, dx, dv]
    >>> # Second half (dx, dv) is variational direction (auto-normalized)
    >>> q0 = np.array([1.0, 0.0, 1.0, 0.0])
    >>> ode = system.get_variational(t0=0, q0=q0, period=1.0, compiled=True)
    >>> ode.integrate(100.0)
    >>> lyap_exp = ode.lyap[-1]  # Final Lyapunov exponent estimate
    """

    @overload
    def __init__(self, f : Callable[[float, np.ndarray, *tuple[Any, ...]], np.ndarray], t0: float, q0: np.ndarray, period: float, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=None, first_step=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):
        pass

    @property
    def t_lyap(self)->np.ndarray:
        """
        Times of variational state renormalization.

        Returns
        -------
        np.ndarray
            Array of times at which the variational perturbation was renormalized.
            These correspond to the times at which Lyapunov exponent estimates
            are computed.
        """
        ...

    @property
    def lyap(self)->np.ndarray:
        """
        Lyapunov exponent estimates at each renormalization time.

        Returns
        -------
        np.ndarray
            Lyapunov exponent values at times t_lyap. A single positive value
            indicates chaotic behavior, negative indicates stability.
        """
        ...

    @property
    def kicks(self)->np.ndarray:
        """
        Stretching factors at each renormalization event.

        Returns
        -------
        np.ndarray
            Growth factors |delta_q| at each t_lyap. Used to compute Lyapunov
            exponents as log(kicks).
        """
        ...

    def copy(self)->VariationalLowLevelODE:
        """
        Create a deep copy of this variational ODE object.

        Returns
        -------
        VariationalLowLevelODE
            A new object with identical configuration, history, and Lyapunov data.
        """
        ...


class OdeResult:
    """
    Encapsulation of a single ODE integration result segment.

    OdeResult represents the solution data from a single integration call (e.g.,
    ode.integrate(), ode.go_to(), or ode.rich_integrate()). Unlike LowLevelODE
    which accumulates all history, OdeResult contains only the newly computed segment.

    OdeResult instances are returned by integration methods and provide the same
    access patterns as LowLevelODE for convenience.

    Parameters
    ----------
    result : OdeResult, optional
        Copy constructor. Create a new OdeResult from an existing one.

    Examples
    --------
    >>> ode = LowLevelODE(f, 0, q0)
    >>> result = ode.integrate(10.0)
    >>> print(result.t)       # Times in this segment
    >>> print(result.success) # Did it complete successfully?
    >>> print(result.message) # Why did it stop?
    """

    def __init__(self, result: OdeResult):...

    @property
    def t(self)->np.ndarray:
        """
        Time steps in this integration segment.

        Returns
        -------
        np.ndarray
            Array of time values at which solutions were stored. Monotonically
            increasing if direction=1, decreasing if direction=-1.
        """
        pass

    @property
    def q(self)->np.ndarray:
        """
        Solution states in this integration segment.

        Returns
        -------
        np.ndarray
            Array of shape (n_steps, n_states) where q[i] is the state at t[i].
            q[i].shape equals the initial condition q0 shape.
        """
        pass

    @property
    def event_map(self)->dict[str, np.ndarray[int]]:
        """
        Event occurrence indices in this segment.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping event names to arrays of step indices where events
            occurred within this segment.
        """
        ...

    @property
    def diverges(self)->bool:
        """
        Check if the solution diverged during this segment.

        Returns
        -------
        bool
            True if integration stopped due to divergence.
        """
        ...

    @property
    def success(self)->bool:
        """
        Check if integration completed successfully.

        Returns
        -------
        bool
            True if the segment completed without divergence or other errors.
            False if integration was aborted or encountered problems.
        """
        pass

    @property
    def runtime(self)->float:
        """
        Wall-clock time spent on this integration segment.

        Returns
        -------
        float
            Time in seconds for this segment's integration.
        """
        pass

    @property
    def message(self)->str:
        """
        Integration completion status message.

        Returns
        -------
        str
            Descriptive message explaining why integration stopped (e.g., "reached
            target time", "divergence detected", "event triggered termination").
        """
        pass

    @property
    def scalar_type(self)->str:
        """
        The numerical precision type used.

        Returns
        -------
        str
            One of "double", "float", "long double", "mpreal".
        """
        ...

    def examine(self)->None:
        """
        Print a summary of this integration result.

        Displays detailed information about the integration including success status,
        runtime, number of steps, events detected, and any relevant diagnostics.
        """

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:
        """
        Get times and states where a specific event occurred in this segment.

        Parameters
        ----------
        event : str
            Name of the event to query.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair (t_event, q_event) containing times and states at event occurrences.
        """
        ...


class OdeSolution(OdeResult):
    """
    ODE result with accurate continuous interpolation capability.

    OdeSolution extends OdeResult by providing callable interpolation to evaluate
    the solution at any time within the integration interval, not just at stored
    step times. The interpolation uses a method specialized to the integration
    algorithm (e.g., Hermite interpolation for RK methods).

    This is returned by rich_integrate() which retains all solver steps to ensure
    accurate interpolation. Use this for smooth visualization or evaluation at
    arbitrary times.

    Parameters
    ----------
    result : OdeSolution, optional
        Copy constructor.

    Examples
    --------
    >>> ode = LowLevelODE(f, 0, q0)
    >>> sol = ode.rich_integrate(10.0)
    >>> # Evaluate at specific times
    >>> q_at_5 = sol(5.0)
    >>> q_at_3_14 = sol(3.14)
    >>> # Evaluate at many times at once
    >>> times = np.linspace(0, 10, 1000)
    >>> solution_values = sol(times)  # Shape: (1000, len(q0))
    """

    def __init__(self, result: OdeSolution):...

    @overload
    def __call__(self, t: float)->np.ndarray:
        """
        Evaluate the interpolated solution at a single time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the solution. Must be within the integration interval.

        Returns
        -------
        np.ndarray
            Solution state at time t with the same shape as the initial condition.
        """
        ...

    @overload
    def __call__(self, t: np.ndarray)->np.ndarray:
        """
        Evaluate the interpolated solution at multiple times.

        Parameters
        ----------
        t : np.ndarray
            Array of times at which to evaluate. Must all be within the integration interval.

        Returns
        -------
        np.ndarray
            Solution states at each time. Shape is (len(t), len(q0)).
            solution[i] is the state at time t[i].
        """
        ...


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


def integrate_all(ode_array: Iterable[LowLevelODE], interval: float, t_eval: Iterable = None, event_options: Iterable[EventOpt] = (), threads=-1, display_progress=False)->None:
    """
    Integrate multiple independent ODE systems in parallel.

    This function provides efficient batch integration of many ODE systems using
    multi-threaded parallelization. All systems are integrated forward by the same
    time interval concurrently.

    Parameters
    ----------
    ode_array : iterable of LowLevelODE
        Collection of independent ODE objects to integrate. Each should be
        initialized but not yet fully integrated.

    interval : float
        Duration of integration for each ODE. Must be positive.

    t_eval : array-like, optional
        Times at which to store solutions (same for all ODEs). If None (default),
        all solver steps are stored.

    event_options : iterable of EventOpt, optional
        Event configuration applied to all ODEs (assumes all have identical events).
        Default is ().

    threads : int, optional
        Number of worker threads.
        - 0 or -1 (default): Use number of CPU threads
        - n >= 1: Use n threads

    display_progress : bool, optional
        If True, print progress information during integration. Default is False.

    Notes
    -----

    All ODEs are advanced synchronously. The function returns when all integrations
    complete.

    Examples
    --------
    Integrate 100 ODEs with different initial conditions in parallel:

    >>> from odepack import *
    >>> t, x, v = symbols('t, x, v')
    >>> system = OdeSystem(ode_sys=[v, -x], t=t, q=[x, v])
    >>> # Create 100 slightly different initial conditions
    >>> perturbations = [np.array([0.01*i, 0.0]) for i in range(100)]
    >>> q0 = np.array([1.0, 0.0])
    >>> odes = [system.get(t0=0, q0=q0 + delta, compiled=True) for delta in perturbations]
    >>> integrate_all(odes, interval=10.0, threads=8)
    >>> # Now each ode has integrated results in ode.t and ode.q
    >>> print(f"First ODE reached t={odes[0].t[-1]:.3f}")
    """
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
        Python functions). Can include RK23, RK45, DOP853, BDF, or VariationalSolver.

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
