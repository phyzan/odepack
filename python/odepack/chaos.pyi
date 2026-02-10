import numpy as np
from typing import Iterable, Callable, Any
from .ode import LowLevelODE
from .events import Event
from .solvers import OdeSolver, Func


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

    rtol, atol : float, optional
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|. Used only when the method is adaptive (e.g., RK45, BDF). Ignored for fixed-step RK4,
            except for estimating the initial step size if stepsize=0.

    min_step : float, optional
        Minimum allowed step size. Default is 0 (no minimum).

    max_step : float, optional
        Maximum allowed step size. Default is None (no limit).

    stepsize : float, optional
        Initial step size estimate. If 0 (default), automatically determined.

    direction : {-1, 1}, optional
        Integration direction. 1 for forward (default), -1 for backward.

    args : tuple, optional
        Extra arguments passed to f and jac. Default is ().

    method : str, optional
        Integration method: "RK23", "RK45" (default), "DOP853", "BDF" or "RK4"

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

    def __init__(self, f: Func, jac: Func, t0: float, q0: np.ndarray, period: float, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, args: Iterable = (), method: str = "RK45", scalar_type: str = "double"):
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
        Relative and absolute tolerance for adaptive step control.
        Error estimate: atol + rtol * |q|. Used for adaptive methods. Ignored for fixed-step RK4 except for initial step size estimation.

    min_step, max_step, stepsize : float, optional
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

    def __init__(self, f : Callable[[float, np.ndarray, *tuple[Any, ...]], np.ndarray], t0: float, q0: np.ndarray, period: float, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=None, stepsize=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):
        ...

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
