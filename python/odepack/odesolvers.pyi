from __future__ import annotations
import numpy as np
from typing import Callable, Iterable, overload, Any

Func = Callable[[float, np.ndarray], np.ndarray]
ObjFunc = Callable[[float, np.ndarray], float]
BoolFunc = Callable[[float, np.ndarray], bool]


class EventOpt:
    '''
    Represents Event options for an event during an ode integration.

    name: Name of the event that the rest of the options are about. Must match the event name passed in an ode.

    max_events: Stop counting events after the specified number of them has been encoutered and stored. If max_events = 0, then all events will be ingored. If max_events = -1, all of them will be stored.

    terminate: If True, then the ode integration stops when max_events have been stored. If max_events = -1 or 0, this parameter is ignored.

    period: Events are only stored after every "period" encouters. If period=1, then all of them are stored. If period=2, then the first one is skipped, the second is stored, the thirs is skipped, etc...

    '''

    def __init__(self, name: str, max_events = -1, terminate = False, period=1):...


class Event:

    @property
    def name(self)->str:...

    @property
    def hides_mask(self)->bool:...

    @property
    def scalar_type(self)->str:...


class PreciseEvent(Event):
    '''
    This class represents an event that might occur during an ODE integration.
    This is expressed as an encoutered zero value of an objective function.
    The solver looks for a sign change of the objective function while integrating,
    and when this happens, the event is accurately determined using a root solver.

    Example
    ----------

    Consider the 2D ode system: dx/dt = y, dy/dt = -x

    This can be expressed as:

    def dq_dt(t, q):
        return np.array([q[1], -q[0]])

    e.g. If an event is defined when x becomes 1, but only when y>0 at the same time, then we can define the event as

    def obj_fun(t, q):
        return q[0]-1

    def when_event(t, q):
        return q[1] > 0

    event = PreciseEvent("Event name", obj_fun, when_event, event_tol=1e-20)

    where event_tol is the accuracy while accurately determining the time of the event occurence.

    '''

    def __init__(self, name: str, when: ObjFunc, direction=0, mask: Func=None, hide_mask=False, event_tol=1e-12, scalar_type: str = "double", __Nsys: int = 0, __Nargs: int = 0):
        '''
        Arguments
        ------------------
        name: Name of the event

        when: objective function (continuous scalar function). A sign change in the function will trigger the exact event occurence, and the time of the event will accurately be determined after solving the equation when(t, q(t)) = 0.

        direction: Direction of zero crossing. 0 means any direction, 1 means positive crossing, -1 means negative crossing.

        mask: If provided, the current ode solution will change value at the time of the event occurence. It must return an array of equal shape as the ode system.

        hide_mask: If True, then the ode solution ONLY at the time of the event occurence will appear as if the provided mask function has not been applied to it.

        event_tol: The numerical tolerance used to determine the event accurately. It will be passed as a parameter in the root finder that solves the equation when(t, q(t)) = 0.

        scalar_type: Numeric type for the event ("double", "float", "long double", or "mpreal").

        __Nsys: Internal parameter for low-level compiled events (system size).

        __Nargs: Internal parameter for low-level compiled events (number of extra arguments).
        '''
        pass

    @property
    def event_tol(self)->float:...


class PeriodicEvent(Event):

    '''
    Similar to the PreciseEvent class, but an event is triggered periodically at times
    t = start + n*period, where n is an integer.
    '''

    def __init__(self, name: str, period: float, start: float=None, mask: Func=None, hide_mask=False, scalar_type: str = "double", __Nsys: int = 0, __Nargs: int = 0):...

    @property
    def period(self)->float:...

    @property
    def start(self)->float:...


class OdeSolver:

    @property
    def t(self)->float:
        '''
        The value of the integration variable of the current step
        '''
        pass

    @property
    def q(self)->np.ndarray:
        '''
        Current state vector
        '''
        pass

    @property
    def stepsize(self)->float:
        '''
        Current stepsize that will be used to estimate the next one.
        '''
        pass

    @property
    def diverges(self)->bool:...

    @property
    def is_dead(self)->bool:...

    @property
    def Nsys(self)->int:
        '''
        The ode system size
        '''
        pass

    def show_state(self, digits: int = 8)->None:...

    def advance(self)->None:
        '''
        Advance the solver by a single adaptive time step
        '''

    def advance_to_event(self)->None:
        '''
        Advance the solver until the next event.
        '''

    def reset(self)->None:
        '''
        Reset the solver to its initial condition
        '''

    def copy(self)->OdeSolver:...

    @property
    def scalar_type(self)->str:...


class RK23(OdeSolver):

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        '''
        f: The ode rhs. It must behave like f(t: float, q: array, *args) -> array

        scalar_type: Numeric type ("double", "float", "long double", or "mpreal").
        '''


class RK45(OdeSolver):

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        '''
        scalar_type: Numeric type ("double", "float", "long double", or "mpreal").
        '''
        pass


class DOP853(OdeSolver):

    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        '''
        scalar_type: Numeric type ("double", "float", "long double", or "mpreal").
        '''
        pass


class BDF(OdeSolver):

    def __init__(self, f: Func, jac: Func, t0: float, q0: np.ndarray, *, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, first_step = 0., direction=1, args: Iterable = (), events: Iterable[Event] = (), scalar_type: str = "double"):
        '''
        scalar_type: Numeric type ("double", "float", "long double", or "mpreal").
        '''
        pass


class LowLevelODE:
    '''
    Class representing an ODE. A LowLevelODE object can dynamically grow every time the user calls methods like .integrate, .go_to. Every time an integration method is called, the object's variables grow accordingly.
    '''

    @overload
    def __init__(self, f: Func, t0: float, q0: np.ndarray, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=None, first_step=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):
        '''
        f: f(t, q, ...)->array. Right hand side of the ODE. Must return an array of equal shape as the initial condition q0.

        t0: Initial value of the integration variable.
        q0: Initial value of the ode solution.
        jac: jac(t, q, ...)->matrix. Function that returns the jacobian matrix of the rhs of the ode. Its matrix elements are Jac[i, j] = df_i/dq_j.
        rtol, atol: Relative and absolute tolerance for the (adative) solver. The solver automatically adapts its stepsize to reduce the local error estimate below atol + rtol*func.
        min_step: minimum stepsize the solver can take.
        max_step: maximum stepsize the solver can take.
        first_step: The initial stepsize the solver will use to estimate the first true stepsize.
        direction: Integration direction (1 for forward, -1 for backward).
        args: additional arguments to be passed in the ode rhs, jacobian and event objective functions.
        events: iterable of Event objects (or any class derived from it).
        method: Integration method used throughout the lifetime of the LowLevelODE object. For stiff problems, use the "BDF" method.
        scalar_type: Numeric type ("double", "float", "long double", or "mpreal").
        '''
        pass

    def solver(self)->OdeSolver:
        '''
        Returns a copy of the underlying OdeSolver in its current state
        '''
        pass

    def integrate(self, interval, *, t_eval : Iterable[float] = None, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeResult:
        '''
        interval: integration interval (must be positive).
        t_eval: Times at which to store the computed solution. If None (default), all steps that the solver encounters will be stored.

        event_options: Optionally provide Event options for each Event in the ode.

        max_prints: Print out the percentage of the ode integration completed. If max_prints=100, then the progress will be printed at 1%, 2%, ... If max_prints = 1000, then the progress will be printed at 0.1%, 0.2%, ... Provide a larger number to diplay the progress with more digits and more frequently (large numbers might affect performance).


        Returns
        --------------
        OdeResult: result encapsulating the time steps stored during the ode integration, based on the provided parameters.

        The LowLevelODE object will also grow to include the time steps and event occurences stored in the OdeResult.
        '''
        pass

    def go_to(self, t, *, t_eval : Iterable[float] = None, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeResult:...

    def rich_integrate(self, interval, *, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeSolution:
        '''
        Similar to .integrate(), but all time steps will be stored in the result.
        This is because they are all required to provide an accurate interpolation in the provided interval, accessible in the OdeSolution object that is returned.
        '''
        pass

    @property
    def t(self)->np.ndarray:...

    @property
    def q(self)->np.ndarray:...

    @property
    def Nsys(self)->int:...

    @property
    def runtime(self)->float:...

    @property
    def diverges(self)->bool:...

    @property
    def is_dead(self)->bool:...
    
    @property
    def event_map(self)->dict[str, np.ndarray[int]]:...

    @property
    def scalar_type(self)->str:...

    def copy(self)->LowLevelODE:...

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:...

    def reset(self)->None:...

    def clear(self)->None:...




class VariationalLowLevelODE(LowLevelODE):
    '''
    Class for variational ODEs (used for Lyapunov exponent calculations).
    '''

    @overload
    def __init__(self, f : Callable[[float, np.ndarray, *tuple[Any, ...]], np.ndarray], t0: float, q0: np.ndarray, period: float, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=None, first_step=0., direction=1, args=(), events: Iterable[Event]=(), method="RK45", scalar_type: str = "double"):...

    @property
    def t_lyap(self)->np.ndarray:...

    @property
    def lyap(self)->np.ndarray:...

    @property
    def kicks(self)->np.ndarray:...

    def copy(self)->VariationalLowLevelODE:...


class OdeResult:

    '''
    An object encapsulating the result returned by the integration of an ode.
    '''

    def __init__(self, result: OdeResult):... #copy constructor

    @property
    def t(self)->np.ndarray:
        '''
        An array with time steps stored during an ode integration, corresponding to integration steps.
        '''
        pass

    @property
    def q(self)->np.ndarray:
        '''
        An array storing the ode solution corresponding to every time step in OdeResult.t

        q[0]: solution at t[0]
        q[1]: solution at t[1]
        ...
        ...

        First axis: time steps.
        For every time step "i", q[i] has the same array-shape as the initial condition provided in the ode.

        '''
        pass

    @property
    def event_map(self)->dict[str, np.ndarray[int]]:
        '''
        A dictionary that maps an event name to the time steps of the event occurence (array of integers), such that

        if

        event_indices = result.event_map["Event name"] #array of integers/indices

        then

        t_events = result.t[event_indices] #array of time steps

        and

        q_events = result.q[event_indices] #solution values at every event occurence

        '''
        pass

    @property
    def diverges(self)->bool:
        '''
        Whether or not the ode stopped because the solution encoutered nan or infinite values.
        '''
        pass

    @property
    def success(self)->bool:
        '''
        Returns True if the ode integration was completed without any problems (e.g. diverging solution).
        '''
        pass

    @property
    def runtime(self)->float:
        '''
        Total time (seconds) that the ode integration lasted.
        '''
        pass

    @property
    def message(self)->str:
        '''
        Cause of the ode integration completion
        '''
        pass

    @property
    def scalar_type(self)->str:...
        

    def examine(self):
        '''
        Prints an overall statement about the ode result.
        '''
        pass

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:
        '''
        event: Name of the event

        Similar to event_map, but directly returns both the time steps and the solution values at every event occurence for the given event.
        '''
        pass


class OdeSolution(OdeResult):

    '''
    Inheriting the propertied of the OdeResult, this class
    also provides an accurate continuous interpolation of values for the ode solution.

    The interpolation method is specialized to provide an accurate result based on the integration method provided in the solver.

    It is returned after a rich_integrate call in an ode object.
    '''

    def __init__(self, result: OdeSolution):... #copy constructor

    @overload
    def __call__(self, t: float)->np.ndarray:...

    @overload
    def __call__(self, t: np.ndarray)->np.ndarray:...


class LowLevelFunction:

    def __init__(self, pointer, input_size: int, output_shape: Iterable[int], Nargs: int, scalar_type: str = "double"):...

    def __call__(self, t: float, q: np.ndarray)->np.ndarray: ...

    @property
    def scalar_type(self)->str:...


def integrate_all(ode_array: Iterable[LowLevelODE], interval: float, t_eval: Iterable = None, event_options: Iterable[EventOpt] = (), threads=-1, display_progress=False)->None:
    '''
    Integrate multiple ODE systems in parallel.

    ode_array: Iterable of LowLevelODE objects to integrate.
    interval: Integration interval.
    t_eval: Times at which to store solutions.
    event_options: Event options for all ODEs.
    threads: Number of threads to use (-1 for auto).
    display_progress: Whether to display progress.
    '''
    ...

def set_mpreal_prec(bits: int)->None:
    '''
    Set the default MPFR precision (in bits) for mpfr::mpreal.
    '''
    ...

def mpreal_prec()->int:
    '''
    Get the default MPFR precision (in bits) for mpfr::mpreal.
    '''
    ...
