from __future__ import annotations
import numpy as np
from typing import Callable, Iterable, overload, Any

Func = Callable[[float, np.ndarray], np.ndarray] #f(t, q, *args) -> q
ObjFunc = Callable[[float, np.ndarray], float] #  f(t, q, *args) -> float
BoolFunc = Callable[[float, np.ndarray], bool] #  f(t, q, *args) -> bool


class Event:

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

    event = Event("Event name", obj_fun, when_event, event_tol=1e-20)

    where event_tol is the accuracy while accurately determining the time of the event occurence.

    '''

    def __init__(self, name: str, when: ObjFunc, dir=0, mask: Func=None, hide_mask=False, event_tol=1e-12):
        '''
        Arguments
        ------------------
        name: Name of the event

        when: objective function (continuous scalar function). A sign change in the function will trigger the exact event occurence, and the time of the event will accurately be determined after solving the equation when(t, q(t)) = 0.

        mask: If provided, the current ode solution will change value at the time of the event occurence. It must return an array of equal shape as the ode system.

        hide_mask: If True, then the ode solution ONLY at the time of the event occurence will appear as if the provided mask function has not been applied to it.

        event_tol: The numerical tolerance used to determine the event accurately. It will be passed as a parameter in the root finder that solves the equation when(t, q(t)) = 0.
        '''
        pass

    @property
    def name(self)->str:...
    
    @property
    def hide_mask(self)->bool:...


class PeriodicEvent(Event):

    '''
    Similar to the Event class, but an event is triggered periodically at times
    t = start + n*period, where n is an integer.
    '''

    def __init__(self, name: str, period: float, start: float=None, mask: Func=None, hide_mask=False):...

    @property
    def period(self)->float:...

    @property
    def start(self)->float:...


class EventOpt:

    '''
    Represents Event options for an event during an ode integration.

    name: Name of the event that the rest of the options are about. Must match the event name passed in an ode.

    max_events: Stop counting events after the specified number of them has been encoutered and stored. If max_events = 0, then all events will be ingored. If max_events = -1, all of them will be stored.

    terminate: If True, then the ode integration stops when max_events have been stored. If max_events = -1 or 0, this parameter is ignored.

    period: Events are only stored after every "period" encouters. If period=1, then all of them are stored. If period=2, then the first one is skipped, the second is stored, the thirs is skipped, etc...


    '''

    def __init__(self, name: str, max_events = -1, terminate = False, period=1):...


class OdeResult:

    '''
    An object encapsulating the result returned by the integration of an ode.
    '''

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

    @overload
    def __call__(self, t: float)->np.ndarray:...

    @overload
    def __call__(self, t: np.ndarray)->np.ndarray:...


class SolverState:

    '''
    Encapsulates the current state of an Ode solver at any time.
    '''

    @property
    def t(self)->float:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event(self)->str:...

    @property
    def diverges(self)->bool:...

    @property
    def is_running(self)->bool:...

    @property
    def is_dead(self)->bool:...

    @property
    def N(self)->int:...

    @property
    def message(self)->str:...

    def show(self):...


class LowLevelFunction:

    def __init__(self, pointer, input_size: int, output_shape: Iterable[int], Nargs: int):...

    def __call__(self, t: float, q: np.ndarray, *args: float)->np.ndarray: ...


class LowLevelODE:

    '''
    Main class representing an ODE. A LowLevelODE object can dynamically grow every time the user calls methods like .integrate, .advance, .go_to. Every time an integration method is called, the object's variables grow accordingly.
    '''

    @overload
    def __init__(self, f: Callable[[float, np.ndarray, *tuple[Any, ...]], np.ndarray], t0: float, q0: np.ndarray, *, jac: Callable = None, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), events: Iterable[Event]=(), method="RK45"):
        '''
        f: f(t, q, ...)->array. Right hand side of the ODE. Must return an array of equal shape as the initial condition q0.

        t0: Initial value of the integration variable.
        q0: Initial value of the ode solution.
        jac: jac(t, q, ...)->matrix. Function that returns the jacobian matrix of the rhs of the ode. Its matrix elements are Jac[i, j] = df_i/dq_j.
        rtol, atol: Relative and absolute tolerance for the (adative) solver. The solver automatically adapts its stepsize to reduce the local error estimate below atol + rtol*func.
        min_step: minimum stepsize the solver can take.
        max_step: maximum stepsize the solver can take.
        first_step: The initial stepsize the solver will use to estimate the first true stepsize.
        args: additional arguments to be passed in the ode rhs, jacobian and event objective functions.
        events: iterable of Event objects (or any class derived from it).
        method: Integration method used throughout the lifetime of the LowLevelODE object. For stiff problems, use the "BDF" method.
        '''
        pass

    @overload
    def __init__(self, ode: LowLevelODE):...

    def integrate(self, interval, *, max_frames=-1, event_options: Iterable[EventOpt] = (), max_prints=0, include_first=False)->OdeResult:
        '''
        interval: integration interval (can take negative values).
        max_frames: The number of frames to be stored in the ode object. This is in case not all time steps are needed, but only e.g 100 of them. In many cases the computer might run out of memory, so it is more efficient to store only a few time steps during the ode integration. These will be (approximately) evenly spread throughout the interval, but do not include the time steps corresponding to event occurences (those are separate).

        event_options: Optionally provide Event options for each Event in the ode.

        max_prints: Print out the percentage of the ode integration completed. If max_prints=100, then the progress will be printed at 1%, 2%, ... If max_prints = 1000, then the progress will be printed at 0.1%, 0.2%, ... Provide a larger number to diplay the progress with more digits and more frequently (large numbers might affect performance).

        include_first: If True, the initial time step at the start of the integration will be included in the result


        Returns
        --------------
        OdeResult: result encapsulating the time steps stored during the ode integration, based on the provided parameters.

        The LowLevelODE object will also grow to include the time steps and event occurences stored in the OdeResult.
        '''
        pass

    def rich_integrate(self, interval, *, event_options: Iterable[EventOpt] = (), max_prints=0)->OdeSolution:
        '''
        Similar to .integrate(), but all time steps will be stored in the result.
        This is because they are all required to provide an accurate interpolation in the provided interval, accessible in the OdeSolution object that is returned.
        '''

    def go_to(self, t, *, max_frames=-1, event_options: Iterable[EventOpt] = (), max_prints=0, include_first=False)->OdeResult:...

    def advance(self)->bool:...

    def state(self)->SolverState:...

    def copy(self)->LowLevelODE:...

    def save_data(self, save_dir: str):...

    def clear(self):...

    def reset(self):...

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:...

    @property
    def dim(self)->int:...

    @property
    def t(self)->np.ndarray:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event_map(self)->dict[str, np.ndarray]:...

    @property
    def runtime(self)->float:...

    @property
    def diverges(self)->bool:...

    @property
    def is_dead(self)->float:...


class VariationalLowLevelODE(LowLevelODE):
    
    @overload
    def __init__(self, f : Callable[[float, np.ndarray, *tuple[Any, ...]], np.ndarray], t0: float, q0: np.ndarray, period: float, *, jac: Callable = None, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), events: list[Event]=None, method="RK45"):...

    @overload
    def __init__(self, ode: VariationalLowLevelODE):...

    @property
    def t_lyap(self)->np.ndarray:...

    @property
    def lyap(self)->np.ndarray:...

    def copy(self)->VariationalLowLevelODE:...


def integrate_all(ode_array: Iterable[LowLevelODE], interval, max_frames=-1, event_options: Iterable[EventOpt] = (), threads=-1, display_progress=False):...
