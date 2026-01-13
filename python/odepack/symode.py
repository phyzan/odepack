from __future__ import annotations
from typing import Iterable, Callable
from .odesolvers import * #type: ignore
from numiphy.lowlevelsupport import *


class SymbolicEvent:
    """Abstract base class for symbolic event definitions.

    Symbolic events define conditions during ODE integration that trigger when a
    specific mathematical expression equals zero. This is an abstract base class
    that must be subclassed (use SymbolicPreciseEvent or SymbolicPeriodicEvent).

    Parameters
    ----------
    name : str
        A descriptive name for the event. Used for identification and reporting.
    mask : Iterable[Expr], optional
        Symbolic expressions representing the new state vector when the event
        triggers, e.g., mask = [x_new, y_new, ...]. If None, no state
        modification occurs at the event (state-neutral event).
    hide_mask : bool, default False
        If True, the mask-modified state is not shown in results at the event.
        Internally the mask is applied for continuation, but the output retains
        the pre-event state.

    Raises
    ------
    NotImplementedError
        If attempting to instantiate SymbolicEvent directly instead of through
        a subclass (SymbolicPreciseEvent or SymbolicPeriodicEvent).

    Notes
    -----
    This class is abstract and should not be instantiated directly. Use one of
    the concrete subclasses:
    - SymbolicPreciseEvent: Triggers when a symbolic expression equals zero
    - SymbolicPeriodicEvent: Triggers at regular time intervals

    See Also
    --------
    SymbolicPreciseEvent : Event triggered by expression zero crossing
    SymbolicPeriodicEvent : Event triggered at periodic time intervals
    OdeSystem : Main class for defining and solving ODE systems with events
    """

    name: str
    mask: Iterable[Expr]
    hide_mask: bool

    def __init__(self, name: str, mask: Iterable[Expr]=None, hide_mask=False):
        if (self.__class__ == SymbolicEvent):
            raise NotImplementedError('The SymbolicEvent class is an abstract base class. It can only be instanciated through a subclass')
        self.name = name
        self.mask = mask
        self.hide_mask = hide_mask

    def __eq__(self, other):
        if type(other) is type(self):
            if other is self:
                return True
            else:
                return (self.name, self.mask, self.hide_mask) == (other.name, other.mask, other.hide_mask)
        return False
    
    @property
    def kwargs(self):
        """Return keyword arguments for event creation.

        Returns
        -------
        dict
            Dictionary containing 'name' and 'hide_mask' keys. Subclasses should
            override this to include additional parameters specific to the event type.
        """
        return {"name": self.name, "hide_mask": self.hide_mask}

    def _toEvent(self, scalar_type, **kwargs):
        """Convert symbolic event to a compiled event object.

        This is an internal method used to create the low-level event object
        that will be used during integration. Subclasses must override this.

        Parameters
        ----------
        scalar_type : str
            The scalar type to use: 'double' (IEEE 754 double precision),
            'float' (IEEE 754 single precision), 'long double' (extended
            precision), or 'mpreal' (arbitrary precision via MPFR library)
        **kwargs
            Additional keyword arguments for event creation

        Returns
        -------
        Event
            A compiled event object (PreciseEvent or PeriodicEvent)

        Notes
        -----
        This method is called internally by OdeSystem when converting symbolic
        events to runtime event objects. Users do not call this directly.
        """
        ...

    def _funcs(self, t, *q, args, scalar_type='double')->dict[str, LowLevelCallable]:
        """Generate low-level callable functions for the event.

        This is an internal method used to compile event functions. Subclasses
        must override this method. Called internally during OdeSystem compilation.

        Parameters
        ----------
        t : Symbol
            The time variable
        *q : Symbol
            The solution variables
        args : Iterable[Symbol]
            System parameters (not time-integrated)
        scalar_type : str, default 'double'
            The scalar type to use for compilation: 'double', 'float',
            'long double', or 'mpreal'

        Returns
        -------
        dict[str, LowLevelCallable]
            Dictionary mapping parameter names to compiled callable objects.
            For PreciseEvent: {"when": callable, "mask": callable or None}
            For PeriodicEvent: {"mask": callable or None}

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented by subclasses

        Notes
        -----
        The mask callable represents the masked state vector (if any), mapping
        (t, q, *args) -> new_q_vector. If no mask is defined, returns None.
        """
        raise NotImplementedError('')
    


class SymbolicPreciseEvent(SymbolicEvent):
    """Symbolic event triggered when an expression equals zero.

    A precise event fires when a mathematical expression crosses zero during ODE
    integration. This is useful for detecting specific conditions like collisions,
    zero-crossings, or equilibrium points. The event detection uses a bracketing
    algorithm to locate the zero-crossing with high precision.

    Parameters
    ----------
    name : str
        A descriptive name for the event (e.g., "collision", "equilibrium")
    event : Expr
        A symbolic expression that triggers the event when it equals zero. This
        expression can depend on the time variable, solution variables, and
        system parameters.
    direction : {-1, 0, 1}, default 0
        Controls which zero-crossings are detected:
        - 0: Detect both upward and downward crossings
        - 1: Detect only upward crossings (event < 0 to event > 0)
        - -1: Detect only downward crossings (event > 0 to event < 0)
    mask : Iterable[Expr], optional
        Symbolic expressions representing the new state vector when the event
        triggers, e.g., mask = [x_new, y_new, ...]. If None, no state
        modification occurs at the event (state-neutral event).
    hide_mask : bool, default False
        If True, the mask-modified state is not shown in results at the event.
        Internally the mask is applied for continuation, but the output retains
        the pre-event state.
    event_tol : float, default 1e-12
        Tolerance for event detection. The integrator will locate the event
        time to within this tolerance.

    Examples
    --------
    Create an event that triggers when a solution variable equals zero:

    >>> from odepack import *
    >>> x, y, t = symbols('x, y, t')
    >>> event = SymbolicPreciseEvent(
    ...     name="zero_crossing",
    ...     event=x,
    ...     direction=1  # Only upward crossings
    ... )

    Create an event with state modification (e.g., bouncing):

    >>> event_with_mask = SymbolicPreciseEvent(
    ...     name="bounce",
    ...     event=x - 1.0,  # When x reaches 1
    ...     mask=[x, -y]    # Reverse y-component (velocity)
    ... )

    See Also
    --------
    SymbolicPeriodicEvent : Event at regular time intervals
    SymbolicEvent : Abstract base class
    """

    def __init__(self, name: str, event: Expr, direction=0, mask: Iterable[Expr]=None, hide_mask=False, event_tol=1e-12):
        SymbolicEvent.__init__(self, name, mask, hide_mask)
        self.event = event
        self.direction = direction
        self.event_tol = event_tol
    
    def __eq__(self, other):
        if type(other) is type(self):
            if other is self:
                return True
            else:
                return (self.event, self.direction, self.event_tol) == (other.event, other.direction, other.event_tol) and SymbolicEvent.__eq__(self, other)
        return False
    
    @property
    def kwargs(self):
        return dict(**SymbolicEvent.kwargs.__get__(self), direction=self.direction, event_tol=self.event_tol)
    
    def _toEvent(self, scalar_type, when, mask=None, **__extra):
        return PreciseEvent(scalar_type=scalar_type, when=when, mask=mask, **self.kwargs, **__extra)
    
    def _funcs(self, t, *q, args, scalar_type='double')->dict[str, LowLevelCallable]:
        ev_code = ScalarLowLevelCallable(self.event, t, q=q, args=args, scalar_type=scalar_type)

        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args, scalar_type=scalar_type)
            return {"when": ev_code, "mask": mask}
        else:
            return {"when": ev_code}


class SymbolicPeriodicEvent(SymbolicEvent):
    """Symbolic event that triggers at regular time intervals.

    A periodic event fires at fixed time intervals during integration. This is
    useful for recording solution snapshots at regular intervals, monitoring
    periodic behavior, or implementing adaptive stepping based on time.

    Parameters
    ----------
    name : str
        A descriptive name for the event (e.g., "snapshot", "output")
    period : float
        The time interval between successive event triggers. Must be positive.
        Triggers occur at times t0 + period, t0 + 2*period, t0 + 3*period, ...
    start : float, optional
        The time of the first event trigger. If None, the first event occurs at
        t0 + period (where t0 is the initial integration time). This parameter
        allows for staggered event times.
    mask : Iterable[Expr], optional
        Symbolic expressions representing the new state vector when the event
        triggers, e.g., mask = [x_new, y_new, ...]. If None, no state
        modification occurs at the event (state-neutral event).
    hide_mask : bool, default False
        If True, the mask-modified state is not shown in results at the event.
        Internally the mask is applied for continuation, but the output retains
        the pre-event state.

    Examples
    --------
    Create an event that triggers every 0.1 time units starting at t=1.0:

    >>> event = SymbolicPeriodicEvent(
    ...     name="regular_output",
    ...     period=0.1,
    ...     start=1.0
    ... )

    Create an event that triggers every 1.0 time units starting from t0+1.0:

    >>> event = SymbolicPeriodicEvent(
    ...     name="snapshots",
    ...     period=1.0,
    ...     start=None  # First event at t0 + period
    ... )

    See Also
    --------
    SymbolicPreciseEvent : Event triggered by expression zero crossing
    SymbolicEvent : Abstract base class
    """

    def __init__(self, name: str, period: float, start = None, mask: Iterable[Expr]=None, hide_mask=False):
        SymbolicEvent.__init__(self, name, mask, hide_mask)
        self.period = period
        self.start = start

    def __eq__(self, other):
        if isinstance(other, SymbolicPeriodicEvent):
            if other is self:
                return True
            elif (self.period, self.start) == (other.period, other.start):
                return SymbolicEvent.__eq__(self, other)
        return False
    
    @property
    def kwargs(self):
        return dict(**SymbolicEvent.kwargs.__get__(self), period=self.period, start=self.start)

    def _toEvent(self, scalar_type, mask=None, **__extra):
        '''
        override
        '''
        return PeriodicEvent(scalar_type=scalar_type, **self.kwargs, mask=mask, **__extra)

    def _funcs(self, t, *q, args, scalar_type='double')->dict[str, LowLevelCallable]:
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args, scalar_type=scalar_type)
            return {"mask": mask}
        else:
            return {}
        



class OdeSystem:
    """High-level interface for defining and solving ODE systems with events.

    OdeSystem is the main class for working with ordinary differential equations.
    It accepts a system definition using numiphy symbolic expressions, automatically
    compiles them to efficient C++ code, and provides interfaces for numerical
    integration with event detection.

    The general ODE form is: dq/dt = f(t, q, *args), where:
    - t is the time variable
    - q is the state vector (length Nsys)
    - args are extra parameters (length Nargs)

    The system can be solved in two ways:
    - Using Python functions (slower but always available)
    - Using compiled C++ functions (50-100x faster, requires compilation)

    Parameters
    ----------
    ode_sys : Iterable[Expr]
        Sequence of symbolic expressions representing dq_i/dt. Must have the same
        length as the q parameter. ode_sys[i] defines the right-hand side of the
        i-th equation.
    t : Symbol
        numiphy Symbol representing time
    q : Iterable[Symbol]
        Sequence of numiphy Symbols representing solution variables
    args : Iterable[Symbol], optional
        Sequence of numiphy Symbols representing system parameters. These can
        appear in ode_sys but are not time-integrated. Multiple LowLevelODE
        instances can be created from the same OdeSystem with different
        parameter values without recompilation. Default is empty tuple.
    events : Iterable[SymbolicEvent], optional
        Sequence of symbolic event definitions (SymbolicPreciseEvent or
        SymbolicPeriodicEvent) that trigger during integration. Default is
        empty tuple.
    module_name : str, optional
        Name for the compiled Python module. If None, a random module name is generated
        for each compilation and the module is stored in a temporary directory (not saved
        to disk). If provided, compiled modules are saved to disk with names like
        'module_name_double', 'module_name_float', etc. for each scalar type.
    directory : str, optional
        Directory where compiled binary modules (.so/.pyd files) will be saved.
        Only used if module_name is not None. If directory is None but module_name
        is provided, modules are saved to the current working directory (os.getcwd()).

    Attributes
    ----------
    Nsys : int
        Number of equations (equal to len(q))
    Nargs : int
        Number of parameters (equal to len(args))
    ode_sys : tuple[Expr, ...]
        The ODE equations as a tuple
    q : tuple[Symbol, ...]
        Solution variables as a tuple
    t : Symbol
        Time variable
    args : tuple[Symbol, ...]
        Parameters as a tuple
    events : tuple[SymbolicEvent, ...]
        Event definitions as a tuple

    Examples
    --------
    Define and solve a Lotka-Volterra predator-prey system:

    >>> from odepack import *
    >>> t, x, y, a, b, c, d = symbols('t, x, y, a, b, c, d')
    >>>
    >>> # Define the ODE system: dx/dt = a*x - b*x*y, dy/dt = c*x*y - d*y
    >>> system = OdeSystem(
    ...     ode_sys=[a*x - b*x*y, c*x*y - d*y],
    ...     t=t,
    ...     q=[x, y],
    ...     args=[a, b, c, d]
    ... )
    >>>
    >>> # Create a solver with initial conditions
    >>> solver = system.get(
    ...     t0=0.0,
    ...     q0=[1.0, 1.0],
    ...     args=(1.0, 0.1, 0.1, 0.4),
    ...     method="RK45",
    ...     compiled=True
    ... )
    >>>
    >>> # Integrate to t=10
    >>> result = solver.integrate(10.0)

    Define a system with events:

    >>> from odepack import *
    >>> t, x, y = symbols('t, x, y')
    >>>
    >>> # Event: detect when x = 1.0
    >>> collision = SymbolicPreciseEvent(
    ...     name="x_equals_one",
    ...     event=x - 1.0,
    ...     direction=0  # Both crossings
    ... )
    >>>
    >>> # Event: record solution every dt=0.1
    >>> sampling = SymbolicPeriodicEvent(
    ...     name="sample",
    ...     period=0.1
    ... )
    >>>
    >>> system = OdeSystem(
    ...     ode_sys=[y, -x],
    ...     t=t,
    ...     q=[x, y],
    ...     events=[collision, sampling]
    ... )
    >>>
    >>> solver = system.get(t0=0.0, q0=[1.0, 0.0])
    >>> result = solver.integrate(10.0)

    See Also
    --------
    VariationalOdeSystem : Augment system with explicit variational equations
    LowLevelODE : Solver object returned by get() method
    SymbolicPreciseEvent : Event triggered by expression zero crossing
    SymbolicPeriodicEvent : Event triggered at time intervals
    """

    _cls_instances: list[OdeSystem] = []

    def __init__(self, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=(), module_name: str=None, directory: str = None):
        self.__directory = directory if directory is not None else os.getcwd()
        self.__module_name = module_name if module_name is not None else 'ode_module'
        self.__nan_dir = directory is None
        self.__nan_modname = module_name is None
        self._pointers_cache = {}
        self._pointers_var_cache = {}
        self._pointers_jac_cache = {}
        self._process_args(ode_sys, t, q, args=args, events=events)
        # Initialize caches for scalar_type-specific low-level functions
    
    def _process_args(self, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
        q = tuple([qi for qi in q])
        ode_sys = tuple(ode_sys)
        args = tuple(args)
        events = tuple(events)

        given = (t,)+q+args
        assert tools.all_different(given)
        odesymbols = []
        for ode in ode_sys:
            for arg in ode.variables:
                if arg not in odesymbols:
                    odesymbols.append(arg)
        if len(ode_sys) != len(q):
            raise ValueError('')
        if t in odesymbols:
            assert len(odesymbols) <= len(given)
        else:
            assert len(odesymbols) <= len(given) - 1
        
        self._cached_expressions = {'odesys': ode_sys, 'jacmat': None, 'odesys_aug': None, 'jacmat_aug': None}
        self._args = args
        self._t = t
        self._q = q
        self._events = events

        for i in range(len(self.__class__._cls_instances)):
            if self.__class__._cls_instances[i] == self:
                obj = self.__class__._cls_instances[i]
                self._pointers_cache = obj._pointers_cache
                self._pointers_var_cache = obj._pointers_var_cache
                self._pointers_jac_cache = obj._pointers_jac_cache
                return
        self.__class__._cls_instances.append(self)

    def __eq__(self, other: OdeSystem):
        if other is self:
            return True
        else:
            return (self.ode_sys, self.args, self.t, self.q, self.events) == (other.ode_sys, other.args, other.t, other.q, other.events)

    # ==================== PROPERTIES ====================

    @property
    def t(self):
        return self._t
    
    @property
    def q(self):
        return self._q
    
    @property
    def events(self):
        return self._events
    
    @property
    def args(self):
        return self._args
    
    @property
    def ode_sys(self)->tuple[Expr, ...]:
        return self._cached_expressions['odesys']
    
    @property
    def jacmat(self)->list[list[Expr]]:
        if self._cached_expressions['jacmat'] is None:
            self._cached_expressions['jacmat'] = self._get_jacmat
        return self._cached_expressions['jacmat']

    @property
    def ode_sys_augmented(self)->tuple[Expr, ...]:
        if self._cached_expressions['odesys_aug'] is None:
            self._cached_expressions['odesys_aug'] = self._get_ode_sys_augmented
        return self._cached_expressions['odesys_aug']
    
    @property
    def jacmat_augmented(self)->list[list[Expr]]:
        if self._cached_expressions['jacmat_aug'] is None:
            self._cached_expressions['jacmat_aug'] = self._get_jacmat_augmented
        return self._cached_expressions['jacmat_aug']

    @property
    def directory(self):
        """Directory where compiled modules are saved."""
        return self.__directory

    @property
    def Nsys(self):
        """Number of equations in the system.

        Returns
        -------
        int
            The number of differential equations (equal to len(q))
        """
        return len(self.ode_sys)

    @property
    def Nargs(self):
        """Number of parameters in the system.

        Returns
        -------
        int
            The number of parameters (equal to len(args))
        """
        return len(self.args)
    
    @cached_property
    def Nptrs(self):
        '''
        Number of required pointers required for a fully compiled ode
        This is the same for both the regular system and the variational one
        '''
        return len(self.lowlevel_callables())

    # ==================== CACHED PROPERTIES ====================

    @cached_property
    def q_augmented(self):
        """Augmented state variables for variational equations."""
        return self.q + tuple([Symbol(f'del_{x}') for x in self.q])

    @cached_property
    def python_funcs(self)->tuple[Callable,...]:
        """Python versions of system functions and event functions.

        Returns
        -------
        tuple[Callable, ...]
            Tuple of callable functions. The first element is the right-hand side
            function, the second is the Jacobian, followed by event functions.
            Each callable has signature (t, q, *args) -> array.

        Notes
        -----
        These are slower than compiled C++ functions but always available and
        don't require compilation. Used when compiled=False.
        """
        llc = self.lowlevel_callables()
        res = [None for _ in llc]
        for i in range(len(res)):
            f = llc[i].to_python_callable().lambdify()
            res[i] = lambda t, q, *args : f(t, q, args)
        return tuple(res)

    @cached_property
    def true_py_events(self):
        """Python event functions for the original (non-variational) system."""
        return self._true_events(compiled=False, variational=False)

    @cached_property
    def true_py_events_var(self):
        """Python event functions for the augmented variational system."""
        return self._true_events(compiled=False, variational=True)

    # ==================== PUBLIC METHODS ====================

    def module_name(self, scalar_type='double', variational=False):
        """Get the module name for a given scalar type and variational mode.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type
        variational : bool, default False
            Whether to get the variational module name

        Returns
        -------
        str
            Module name with format: module_name_[var_]scalar_type
        """
        prefix = f"{self.__module_name}_"
        suffix = f"{'var_' if variational else ''}{scalar_type.replace(' ', '_')}"
        return prefix + suffix
    
    def code(self, scalar_type='double', variational=False)->str:
        """Generate C++ source code for the ODE system.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to use in the generated code. Can be 'double'
            (IEEE 754 double precision), 'float' (IEEE 754 single precision),
            'long double' (extended precision), or 'mpreal' (arbitrary precision
            via MPFR library).
        variational : bool, default False
            If True, generate code for the augmented system with variational equations.
            If False, generate code for the original system.

        Returns
        -------
        str
            Complete C++ source code implementing the ODE right-hand side,
            Jacobian, and event functions for the specified scalar type.

        Notes
        -----
        This code is automatically generated from the symbolic expressions.
        The generated code includes function pointers that can be called from
        Python after compilation.
        """
        return generate_cpp_code(self.lowlevel_callables(scalar_type=scalar_type, variational=variational), self.module_name(scalar_type=scalar_type, variational=variational))

    def compile(self, scalar_type='double', variational=False, start_from=0)->tuple:
        """Compile the ODE system to a C++ module.

        Generates C++ source code and compiles it to a Python extension module.
        The compiled module provides fast function pointers for the right-hand
        side, Jacobian, and event functions. This typically provides 50-100x
        speedup compared to Python functions.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double' (IEEE 754 double
            precision), 'float' (IEEE 754 single precision), 'long double'
            (extended precision), or 'mpreal' (arbitrary precision via MPFR).
        variational : bool, default False
            If True, compile the augmented system with variational equations
            (2*Nsys equations total). If False, compile the original system (Nsys equations).

        Returns
        -------
        tuple of function pointers that can be passed to LowLevelODE.
            The first element is the ODE right-hand side pointer, the second
            is the Jacobian pointer, followed by event function pointers.

        Notes
        -----
        This method is automatically called by get() and get_variational() with
        compiled=True.

        Module caching behavior:
        - If module_name was None during OdeSystem creation, the compiled module
          is stored in a temporary directory (not saved to disk).
        - If module_name was provided, the binary module (.so/.pyd) is saved to the
          directory specified (or current working directory if directory was None).
          The binary is named 'module_name_<scalar_type>' for each scalar type.
        """
        result = compile_funcs(self.lowlevel_callables(scalar_type=scalar_type, variational=variational)[start_from:], None if self.__nan_dir else self.directory, None if self.__nan_modname else self.module_name(scalar_type=scalar_type, variational=variational))
        return result

    def ode_to_compile(self, scalar_type='double', variational=False)->TensorLowLevelCallable:
        """Get the ODE right-hand side ready for compilation.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            If True, compile the augmented variational system

        Returns
        -------
        TensorLowLevelCallable
            Low-level representation of the ODE right-hand side
        """
        q, odesys, _ = self._ode_data(variational=variational)
        return TensorLowLevelCallable(array=odesys, t=self.t, q=q, scalar_type=scalar_type, args=self.args)

    def jacobian_to_compile(self, scalar_type='double', variational=False, layout = 'F')->TensorLowLevelCallable:
        """Get the Jacobian matrix ready for compilation.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            If True, compile the augmented variational system Jacobian

        Returns
        -------
        TensorLowLevelCallable
            Low-level representation of the Jacobian matrix
        """
        q, _, jacmat = self._ode_data(variational=variational)
        if layout == 'F':
            jacmat = [[jacmat[j][i] for j in range(len(jacmat))] for i in range(len(jacmat[0]))]
        return TensorLowLevelCallable(array=jacmat, t=self.t, q=q, scalar_type=scalar_type, args=self.args)

    def true_compiled_events(self, scalar_type='double', variational=False):
        """Create compiled Event objects from symbolic event definitions.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            If True, create variational events

        Returns
        -------
        list[Event]
            List of Event objects using compiled function pointers
        """
        return self._true_events(compiled=True, scalar_type=scalar_type, variational=variational)

    def lowlevel_callables(self, scalar_type='double', variational=False)->tuple[LowLevelCallable, ...]:
        """Get all low-level callables for the ODE system (RHS, Jacobian, events).

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            If True, include variational system callables

        Returns
        -------
        tuple[LowLevelCallable, ...]
            Tuple of (ode_rhs, jacobian, *event_callables) ready for compilation
        """
        event_objs: list[LowLevelCallable] = []
        for event_dict in self._event_data(scalar_type=scalar_type, variational=variational):
            for param_name, lowlevel_callable in event_dict.items():
                event_objs.append(lowlevel_callable)
        return (self.ode_to_compile(scalar_type=scalar_type, variational=variational), self.jacobian_to_compile(scalar_type=scalar_type, variational=variational), *event_objs)
    
    def override_odesys(self, odesys: Iterable[Expr]):
        self._cached_expressions['odesys'] = tuple(odesys)
        return
    
    def override_jacmat(self, jacmat: list[list[Expr]]):
        self._cached_expressions['jacmat'] = jacmat
        return
    
    def override_variational_odesys(self, variational_odesys: Iterable[Expr]):
        self._cached_expressions['odesys_aug'] = tuple(variational_odesys)
        return
    
    def override_variational_jacmat(self, jacmat: list[list[Expr]]):
        self._cached_expressions['jacmat_aug'] = jacmat
        return
    
    def override_pointers(self, pointers: dict[str, tuple[Pointer, ...]]):
        for scalar_type, ptrs in pointers.items():
            self._pointers(scalar_type=scalar_type, variational=False, first=ptrs)
        return
    
    def override_varsys_pointers(self, pointers: dict[str, tuple[Pointer, ...]]):
        for scalar_type, ptrs in pointers.items():
            self._pointers(scalar_type=scalar_type, variational=True, first=ptrs)
        return
    
    def get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", compiled=True, scalar_type='double')->LowLevelODE:
        """Create a solver instance for integrating the ODE system.

        Parameters
        ----------
        t0 : float
            Initial time
        q0 : np.ndarray
            Initial state vector. Must have length equal to Nsys.
        rtol : float, default 1e-6
            Relative tolerance for the integrator
        atol : float, default 1e-12
            Absolute tolerance for the integrator
        min_step : float, default 0.
            Minimum allowed step size (0. means no limit)
        max_step : float, default np.inf
            Maximum allowed step size
        first_step : float, default 0.
            Suggested first step size (0. means automatic)
        direction : {-1, 1}, default 1
            Integration direction: 1 for forward, -1 for backward in time
        args : tuple, default ()
            Parameter values for the system. Must have length equal to Nargs.
        method : str, default "RK45"
            Integration method. Options: "RK23", "RK45", "DOP853", "BDF"
        compiled : bool, default True
            Use compiled C++ functions (much faster). If False, uses Python
            functions (always available, no compilation needed).
        scalar_type : str, default 'double'
            Scalar type for computation:
            - 'double': IEEE 754 double precision (Python's default float)
            - 'float': IEEE 754 single precision
            - 'long double': Extended precision (80-bit or 128-bit)
            - 'mpreal': Arbitrary precision via MPFR library

        Returns
        -------
        LowLevelODE
            A solver instance ready for integration. Call integrate(), go_to(),
            or rich_integrate() to perform integration.

        Raises
        ------
        ValueError
            If q0 length doesn't match Nsys or args length doesn't match Nargs

        See Also
        --------
        LowLevelODE : Returned solver object with integration methods
        get_variational : Create solver for variational equations with Lyapunov tracking
        """
        return self._get(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, method=method, period=None, compiled=compiled, scalar_type=scalar_type)

    def get_variational(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", compiled=True, scalar_type='double')->VariationalLowLevelODE:
        """Create a solver for the variational equations of the ODE system.

        This method creates a solver that tracks both the solution and its
        sensitivity to initial conditions (variational equations). The system
        is augmented from Nsys to 2*Nsys equations, with the second half tracking
        perturbations. This is useful for computing Lyapunov exponents or
        understanding solution stability.

        The variational state vector is automatically normalized to unit length
        at initialization and periodically during integration based on the
        specified period.

        Parameters
        ----------
        t0 : float
            Initial time
        q0 : np.ndarray
            Initial state vector for the augmented system. Must have length 2*Nsys.
            The first Nsys elements are the initial conditions for the base system.
            The last Nsys elements specify the direction of the variational vector
            (does not need to be normalized; automatic normalization is applied).
        period : float
            Renormalization period for the variational state. The variational
            part is normalized to unit length every 'period' time units to
            prevent numerical overflow. Used for Lyapunov exponent calculation.
        rtol : float, default 1e-6
            Relative tolerance for the integrator
        atol : float, default 1e-12
            Absolute tolerance for the integrator
        min_step : float, default 0.
            Minimum allowed step size
        max_step : float, default np.inf
            Maximum allowed step size
        first_step : float, default 0.
            Suggested first step size
        direction : {-1, 1}, default 1
            Integration direction: 1 for forward, -1 for backward
        args : tuple, default ()
            Parameter values. Must have length equal to Nargs.
        method : str, default "RK45"
            Integration method: "RK23", "RK45", "DOP853", "BDF"
        compiled : bool, default True
            Use compiled C++ functions (much faster)
        scalar_type : str, default 'double'
            Scalar type for computation:
            - 'double': IEEE 754 double precision
            - 'float': IEEE 754 single precision
            - 'long double': Extended precision
            - 'mpreal': Arbitrary precision via MPFR library

        Returns
        -------
        VariationalLowLevelODE
            A solver for the augmented system (base + variational equations).
            The returned solver has 2*Nsys equations.

        Notes
        -----
        The returned solver has 2*Nsys equations: the first Nsys are the original
        equations, the last Nsys are the variational equations. Access them with:
        - solver.q[:Nsys] : original solution
        - solver.q[Nsys:] : variational part (normalized periodically)

        See Also
        --------
        VariationalLowLevelODE : Returned solver object
        VariationalOdeSystem : Helper to create systems with explicit variational equations
        """
        return self._get(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, method=method, period=period, compiled=compiled, scalar_type=scalar_type)
    
    def get_var_solver(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", compiled=True, scalar_type='double'):
        """Create a low-level VariationalSolver for step-by-step integration.

        This method creates a VariationalSolver (low-level iterator) for variational
        equations, bypassing the history-accumulating wrapper. Unlike get_variational()
        which returns VariationalLowLevelODE (high-level with history), this returns
        the underlying VariationalSolver that maintains only current state.

        All events defined in the OdeSystem are ignored. The solver only includes
        the automatic normalization event required for Lyapunov exponent calculation.

        The variational system integrates 2*Nsys equations: the first Nsys equations
        are the original system, and the last Nsys are the linearized perturbation
        equations. The perturbation is automatically normalized at initialization and
        periodically during integration.

        Parameters
        ----------
        t0 : float
            Initial time
        q0 : np.ndarray
            Initial state vector for the augmented system. Must have length 2*Nsys.
            The first Nsys elements are the initial conditions for the base system.
            The last Nsys elements specify the direction of the variational vector
            (does not need to be normalized; automatic normalization is applied).
        period : float
            Renormalization period for the variational state. The variational
            part is normalized to unit length every 'period' time units to
            prevent numerical overflow. Must be positive. Used for Lyapunov
            exponent calculation.
        rtol : float, default 1e-6
            Relative tolerance for the integrator
        atol : float, default 1e-12
            Absolute tolerance for the integrator
        min_step : float, default 0.
            Minimum allowed step size
        max_step : float, default np.inf
            Maximum allowed step size
        first_step : float, default 0.
            Suggested first step size
        direction : {-1, 1}, default 1
            Integration direction: 1 for forward, -1 for backward
        args : tuple, default ()
            Parameter values. Must have length equal to Nargs.
        method : str, default "RK45"
            Integration method: "RK23", "RK45", "DOP853", "BDF"
        compiled : bool, default True
            Use compiled C++ functions (much faster). If False, uses Python functions.
        scalar_type : str, default 'double'
            Scalar type for computation:
            - 'double': IEEE 754 double precision
            - 'float': IEEE 754 single precision
            - 'long double': Extended precision
            - 'mpreal': Arbitrary precision via MPFR library

        Returns
        -------
        VariationalSolver
            A low-level solver for step-by-step integration. Access current state
            via solver.t, solver.q. Advance with solver.advance(). Access Lyapunov
            metrics via solver.logksi, solver.lyap, solver.t_lyap, solver.delta_s.

        Raises
        ------
        ValueError
            If q0 length is not 2*Nsys
            If period is not positive
            If args length doesn't match Nargs

        Notes
        -----
        This is a low-level method for advanced users who need direct access to the
        solver iterator without history accumulation. For typical use cases, prefer
        get_variational() which provides a higher-level interface with full integration
        history.

        The returned VariationalSolver maintains only current state (t, q, logksi, lyap,
        etc.) and does not accumulate integration history. Use this when you need
        fine-grained control over integration or want to minimize memory usage.

        Examples
        --------
        Create a low-level variational solver:

        >>> from odepack import *
        >>> t, x, v = symbols('t, x, v')
        >>> system = OdeSystem(ode_sys=[v, -x], t=t, q=[x, v])
        >>> # Initial state: [x0, v0, dx, dv]
        >>> q0 = np.array([1.0, 0.0, 1.0, 0.0])
        >>> solver = system.get_var_solver(
        ...     t0=0, q0=q0, period=1.0, compiled=True
        ... )
        >>> # Step through integration manually
        >>> for _ in range(100):
        ...     solver.advance()
        >>> print(f"t={solver.t:.3f}, Lyapunov={solver.lyap:.6f}")

        See Also
        --------
        get_variational : High-level method returning VariationalLowLevelODE with history
        VariationalSolver : The returned solver class
        VariationalLowLevelODE : High-level wrapper with history accumulation
        """
        if compiled:
            f, jac = self._pointers(scalar_type=scalar_type, variational=True)[:2]
        else:
            f, jac = self._var_odefunc, self._var_jac

        if len(q0) != 2*self.Nsys:
            raise ValueError(f"Invalid length of initial state vector : {len(q0)} instead of {2*self.Nsys}")
        if period <= 0:
            raise ValueError("Renormalization period must be positive")
        if len(args) != len(self.args):
            raise ValueError(f"The provided number of args must be {len(self.args), not {len(args)}}")

        return VariationalSolver(f=f, jac=jac, t0=t0, q0=q0, period=period, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, method=method, scalar_type=scalar_type)

    def lowlevel_odefunc(self, scalar_type='double'):
        """Get the compiled ODE right-hand side function.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to use

        Returns
        -------
        LowLevelFunction
            A low-level callable wrapper around the compiled right-hand side.
            Can be called as f(t, q, *args) with shape (Nsys,) -> (Nsys,)
        """
        return self._lowlevel_func(scalar_type=scalar_type, func='rhs')

    def lowlevel_jac(self, scalar_type='double'):
        """Get the compiled Jacobian function.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to use

        Returns
        -------
        LowLevelFunction
            A low-level callable wrapper around the compiled Jacobian.
            Can be called as jac(t, q, *args) with output shape (Nsys, Nsys)
        """
        return self._lowlevel_func(scalar_type=scalar_type, func='jac')

    def odefunc(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        """Evaluate the ODE right-hand side using Python functions.

        This method evaluates the system dq/dt using Python (not compiled).
        It's slower than the compiled version but always available and good
        for testing or when compilation is not feasible.

        Parameters
        ----------
        t : float
            Current time
        q : np.ndarray
            Current state vector, shape (Nsys,)
        *args : float
            Parameter values, must match the number of system parameters

        Returns
        -------
        np.ndarray
            The right-hand side dq/dt, shape (Nsys,)
        """
        return self._odefunc(t, q, *args)

    def jac(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        """Evaluate the Jacobian matrix using Python functions.

        This method computes the Jacobian J[i,j] = d(dq_i/dt)/dq_j using
        Python (not compiled). It's slower than the compiled version but always
        available for testing.

        Parameters
        ----------
        t : float
            Current time
        q : np.ndarray
            Current state vector, shape (Nsys,)
        *args : float
            Parameter values, must match the number of system parameters

        Returns
        -------
        np.ndarray
            The Jacobian matrix, shape (Nsys, Nsys)
        """
        return self._jac(t, q, *args)

    # ==================== PRIVATE METHODS ====================

    @cached_property
    def _get_jacmat(self):
        """Jacobian matrix of the ODE system.

        Returns
        -------
        list[list[Expr]]
            A 2D list where jacmat[i][j] = d(ode_sys[i])/dq[j]. This is the
            symbolic Jacobian matrix before compilation.

        Notes
        -----
        The Jacobian is used by implicit integrators (like BDF) to improve
        convergence. It is automatically compiled when needed.
        """
        array, symbols = self.ode_sys, self.q
        matrix = [[None for i in range(self.Nsys)] for j in range(self.Nsys)]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    @cached_property
    def _get_ode_sys_augmented(self):
        """ODE system augmented with variational equations."""
        var_odesys = []
        q, delq = self.q_augmented[:self.Nsys], self.q_augmented[self.Nsys:]
        for i in range(self.Nsys):
            var_odesys.append(sum([self.ode_sys[i].diff(q[j])*delq[j] for j in range(self.Nsys)]))

        full_sys = self.ode_sys + tuple(var_odesys)
        return full_sys

    @cached_property
    def _get_jacmat_augmented(self):
        """Jacobian matrix of the augmented variational system."""
        array, symbols = self.ode_sys_augmented, self.q_augmented
        matrix = [[None for i in range(self.Nsys*2)] for j in range(self.Nsys*2)]
        for i in range(2*self.Nsys):
            for j in range(2*self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    @cached_property
    def _event_obj_map(self):
        """Mapping from event names to event objects."""
        return {event.name: event for event in self.events}

    @cached_property
    def _odefunc(self):
        """Python-compiled ODE right-hand side function."""
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.ode_sys, "numpy", self.t, q=self.q, **kwargs)

    @cached_property
    def _jac(self):
        """Python-compiled Jacobian function."""
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.jacmat, "numpy", self.t, q=self.q, **kwargs)

    @cached_property
    def _var_odefunc(self):
        """Python-compiled augmented ODE right-hand side function."""
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.ode_sys_augmented, "numpy", self.t, q=self.q_augmented, **kwargs)

    @cached_property
    def _var_jac(self):
        """Python-compiled augmented Jacobian function."""
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.jacmat_augmented, "numpy", self.t, q=self.q_augmented, **kwargs)
    
    def _ode_data(self, variational: bool):
        """Get the ODE system, state variables, and Jacobian matrix for compilation.

        Parameters
        ----------
        variational : bool
            If True, returns augmented system with variational equations.
            If False, returns original system.

        Returns
        -------
        tuple[list, tuple, list]
            (q_vars, ode_system, jacobian_matrix)
        """
        if variational:
            return self.q_augmented, self.ode_sys_augmented, self.jacmat_augmented
        else:
            return self.q, self.ode_sys, self.jacmat

    def _event_data(self, scalar_type='double', variational=False)->tuple[dict[str, LowLevelCallable], ...]:
        """Compile event function callables for the specified scalar type and variational mode.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            Whether to compile variational events

        Returns
        -------
        tuple[dict[str, LowLevelCallable], ...]
            Tuple of dictionaries, one per event, mapping parameter names to LowLevelCallables
        """
        res = ()
        q = self.q_augmented if variational else self.q
        for ev in self.events:
            res += (ev._funcs(self.t, *q, args=self.args, scalar_type=scalar_type),)
        return res

    def _pointers(self, scalar_type='double', variational=False, first = ())->tuple[Pointer, ...]:
        """Get or compile function pointers for the specified scalar type and variational mode.

        Tries to load cached compiled pointers. If not found, compiles the code and caches
        the resulting function pointers.

        Parameters
        ----------
        scalar_type : str, default 'double'
            The scalar type to compile for. Can be 'double', 'float',
            'long double', or 'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            Whether to compile variational code

        Returns
        -------
        tuple of function pointers that can be passed to C++ integrators
        """
        cache = self._pointers_var_cache if variational else self._pointers_cache
        if scalar_type not in cache:
            try:
                default_ptrs = tools.import_lowlevel_module(self.directory, self.module_name(scalar_type=scalar_type, variational=variational)).pointers()
            except ImportError:
                default_ptrs = self.compile(scalar_type=scalar_type, variational=variational)
            cache[scalar_type] = tuple(first) + default_ptrs[len(first):]
        else:
            cache[scalar_type] = tuple(first) + cache[scalar_type][len(first):]
        return cache[scalar_type]

    def _true_events(self, compiled=True, scalar_type='double', variational=False):
        """Create Event objects from symbolic event definitions.

        Parameters
        ----------
        compiled : bool, default True
            Use compiled C++ function pointers. If False, use Python functions.
        scalar_type : str, default 'double'
            The scalar type to use. Can be 'double', 'float', 'long double', or
            'mpreal' (arbitrary precision via MPFR)
        variational : bool, default False
            Whether to create variational events

        Returns
        -------
        list[Event]
            List of Event objects ready to use with ODE solvers
        """
        res = []
        if compiled:
            items = self._pointers(scalar_type=scalar_type, variational=variational)
        else:
            items = self.python_funcs
        i=0
        for event, event_dict in zip(self.events, self._event_data(scalar_type=scalar_type, variational=variational)):
            extra_kwargs = {"__Nsys": self.Nsys, "__Nargs": self.Nargs}
            for param_name, lowlevel_callable in event_dict.items():
                extra_kwargs[param_name] = items[i+2]
                i+=1
            res.append(event._toEvent(scalar_type=scalar_type, **extra_kwargs))
        return res

    def _get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", period=None, compiled=True, scalar_type='double'):
        """Internal method to create a solver instance.

        Parameters
        ----------
        t0 : float
            Initial time
        q0 : np.ndarray
            Initial state vector
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        min_step : float
            Minimum step size
        max_step : float
            Maximum step size
        first_step : float
            Suggested first step size
        direction : {-1, 1}
            Integration direction
        args : tuple
            Parameter values
        method : str
            Integration method
        period : float, optional
            Renormalization period for variational equations. If provided, creates
            a VariationalLowLevelODE instead of LowLevelODE.
        compiled : bool
            Whether to use compiled code
        scalar_type : str
            The scalar type to use. Can be 'double', 'float', 'long double', or
            'mpreal' (arbitrary precision via MPFR)

        Returns
        -------
        LowLevelODE or VariationalLowLevelODE
            Solver instance
        """
        variational = period is not None
        factor = 1 if not variational else 2
        if len(q0) != self.Nsys*factor:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {factor*self.Nsys}")
        elif len(args) != self.Nargs:
            raise ValueError(f"The number of args provided is {len(args)} instead of {self.Nargs}")
        if compiled:
            pointers = self._pointers(scalar_type=scalar_type, variational=variational)
            events = self.true_compiled_events(scalar_type=scalar_type, variational=variational)
            f, jac = pointers[:2]
        else:
            events = self.true_py_events if not variational else self.true_py_events_var
            f = self._odefunc if not variational else self._var_odefunc
            jac = self._jac if not variational else self._var_jac
        kwargs = dict(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
        if not variational:
            getter = LowLevelODE
        else:
            kwargs['period'] = period
            getter = VariationalLowLevelODE
        return getter(f=f, jac=jac, scalar_type=scalar_type, **kwargs)

    def _lowlevel_func(self, scalar_type, func, variational=False)->LowLevelFunction:
        """Create a low-level callable wrapper for ODE or Jacobian functions.

        Parameters
        ----------
        scalar_type : str
            The scalar type to use. Can be 'double', 'float', 'long double', or
            'mpreal' (arbitrary precision via MPFR)
        func : {'rhs', 'jac'}
            Which function to wrap: 'rhs' for ODE right-hand side, 'jac' for Jacobian
        variational : bool, default False
            Whether to wrap the variational (augmented) function

        Returns
        -------
        LowLevelFunction
            A callable wrapper around compiled function pointers
        """
        factor = 1 if not variational else 2
        if func == 'rhs':
            idx = 0
            shape = [self.Nsys*factor]
        elif func == 'jac':
            if (scalar_type, variational) not in self._pointers_jac_cache:
                self._pointers_jac_cache[(scalar_type, variational)] = compile_funcs([self.jacobian_to_compile(scalar_type=scalar_type, variational=variational, layout='C')])[0]
            return LowLevelFunction(pointer=self._pointers_jac_cache[(scalar_type, variational)], input_size=factor*self.Nsys, output_shape=[self.Nsys*factor, self.Nsys*factor], Nargs=self.Nargs, scalar_type=scalar_type)
        else:
            raise ValueError('')
        p = self._pointers(scalar_type=scalar_type, variational=variational)[idx]
        return LowLevelFunction(pointer=p, input_size=factor*self.Nsys, output_shape=shape, Nargs=self.Nargs, scalar_type=scalar_type)


def HamiltonianSystem2D(V: Expr, t: Symbol, x, y, px, py, args = (), events=()):
    """Create a 2D Hamiltonian ODE system from a potential function.

    Constructs a 4-equation ODE system for a 2D Hamiltonian system with
    canonical coordinates (x, y, px, py) where px and py are conjugate momenta.

    The system is derived from Hamilton's equations:
    - dx/dt = dH/dpx = px
    - dy/dt = dH/dpy = py
    - dpx/dt = -dH/dx = -dV/dx
    - dpy/dt = -dH/dy = -dV/dy

    where H = (px^2 + py^2)/2 + V(x,y) is the total energy.

    Parameters
    ----------
    V : Expr
        The potential function V(x, y, ...) as a SymPy expression.
        Can depend on spatial coordinates (x, y) and parameters.
    t : Symbol
        SymPy Symbol for time
    x : Symbol
        Position variable (x-coordinate)
    y : Symbol
        Position variable (y-coordinate)
    px : Symbol
        Conjugate momentum to x
    py : Symbol
        Conjugate momentum to y
    args : Iterable[Symbol], optional
        System parameters that appear in V(x, y, *args). Default is ().
    events : Iterable[SymbolicEvent], optional
        Event definitions. Default is ().

    Returns
    -------
    OdeSystem
        A 4-equation ODE system for (x, y, px, py) following Hamilton's
        equations. Use system.get() to create a solver.

    Examples
    --------
    Create a 2D harmonic oscillator with potential V = x^2/2 + y^2/2:

    >>> from odepack import *
    >>> t, x, y, px, py = symbols('t, x, y, px, py')
    >>> V = (x**2 + y**2) / 2
    >>>
    >>> system = HamiltonianSystem2D(V, t, x, y, px, py)
    >>> solver = system.get(t0=0, q0=[1.0, 1.0, 0.0, 0.0], method="RK45")
    >>> result = solver.integrate(10.0)

    Create a 2D potential well with parameter depth a:

    >>> t, x, y, px, py, a = symbols('t, x, y, px, py, a')
    >>> V = x**2 * y**2 - a * x**2 - a * y**2
    >>>
    >>> system = HamiltonianSystem2D(V, t, x, y, px, py, args=[a])
    >>> solver = system.get(
    ...     t0=0, q0=[0.5, 0.5, 1.0, 1.0],
    ...     args=(1.0,),  # Depth parameter value
    ...     method="DOP853"
    ... )
    >>> result = solver.integrate(10.0)

    See Also
    --------
    HamiltonianVariationalSystem2D : Hamiltonian system with variational equations
    VariationalOdeSystem : General variational system constructor
    OdeSystem : Base ODE system class
    """
    q = [x, y, px, py]
    f = [px, py] + [-V.diff(x), -V.diff(y)]
    return OdeSystem(f, t, q, args=args, events=events)
