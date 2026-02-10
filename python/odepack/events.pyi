from __future__ import annotations
import numpy as np
from typing import Callable

ObjFunc = Callable[[float, np.ndarray], float]
Func = Callable[[float, np.ndarray], np.ndarray]

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
        ...

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
