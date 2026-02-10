import numpy as np


class OdeResult:
    """
    Encapsulation of a single ODE integration result segment.

    OdeResult represents the solution data from a single integration call (e.g.,
    ode.integrate(), ode.integrate_until(), or ode.rich_integrate()). Unlike LowLevelODE
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
        ...

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
        ...

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
        ...

    @property
    def runtime(self)->float:
        """
        Wall-clock time spent on this integration segment.

        Returns
        -------
        float
            Time in seconds for this segment's integration.
        """
        ...

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
