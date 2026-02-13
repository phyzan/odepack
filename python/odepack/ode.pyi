import numpy as np
from typing import Iterable, Callable
from .solvers import OdeSolver, Event, EventOpt, Func
from .oderesult import OdeResult, OdeSolution



class LowLevelODE:
    """
    Container for an ODE problem with dynamic solution history accumulation.

    LowLevelODE is a high-level wrapper around OdeSolver that accumulates and manages
    the complete integration history. Unlike OdeSolver (which focuses on step-by-step
    iteration), LowLevelODE provides convenient methods to integrate over intervals
    and automatically stores all results.

    The object grows dynamically as integration methods are called. Each call to
    integrate(), integrate_until(), or rich_integrate() appends new results to the internal
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

    method : str, optional
        Integration method. One of:
        - "RK23" (fastest, lowest accuracy)
        - "RK45" (recommended default)
        - "DOP853" (highest accuracy)
        - "BDF" (for stiff problems, requires jac)
        - "RK4" (fixed step size Runge-Kutta 4)
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

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, jac: Callable = None, rtol=1e-6, atol=1e-12, min_step=0., max_step=None, stepsize=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):
        ...

    def rhs(self, t: float, q: np.ndarray)->np.ndarray:
        """
        Evaluate the right-hand side function f(t, q) at a given time and state, using the extra args
            provided at initialization.

        Parameters
        ----------
        t : float
            Time at which to evaluate the RHS.

        q : np.ndarray
            State vector at which to evaluate the RHS.

        Returns
        -------
        np.ndarray
            The result of f(t, q), an array of the same shape as q.
        """
        ...

    def jac(self, t: float, q: np.ndarray)->np.ndarray:
        """
        Evaluate the Jacobian matrix at a given time and state, using the extra args
            provided at initialization.

        Parameters
        ----------
        t : float
            Time at which to evaluate the Jacobian.

        q : np.ndarray
            State vector at which to evaluate the Jacobian.

        Returns
        -------
        np.ndarray
            The Jacobian matrix evaluated at (t, q). Shape should be (len(q), len(q)).
            If no Jacobian was provided at initialization, finite difference approximation is used.
        """
        ...

    def solver(self)->OdeSolver:
        """
        Get a copy of the current underlying OdeSolver.

        Returns
        -------
        OdeSolver
            A copy of the internal solver in its current state. Modifications to the
            returned solver do not affect this LowLevelODE object.
        """
        ...

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
        ...

    def integrate_until(self, t, *, t_eval : Iterable[float] = None, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeResult:
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
        ...

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
            Wall-clock time in seconds for all integrate/integrate_until/rich_integrate calls.
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
