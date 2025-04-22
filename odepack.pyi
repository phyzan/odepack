import numpy as np
from typing import Callable, Iterable

Func = Callable[[float, np.ndarray], np.ndarray] #f(t, q, *args) -> q
ObjFunc = Callable[[float, np.ndarray], float] #  f(t, q, *args) -> float
BoolFunc = Callable[[float, np.ndarray], bool] #  f(t, q, *args) -> bool


class AnyEvent:
    
    @property
    def name(self)->str:...
    
    @property
    def mask(self)->Func:...

    @property
    def hide_mask(self)->BoolFunc:...


class Event(AnyEvent):

    def __init__(self, name: str, when: ObjFunc, check_if: BoolFunc=None, mask: Func=None, hide_mask=False, event_tol=1e-12):...

    @property
    def when(self)->ObjFunc:...

    @property
    def check_if(self)->BoolFunc:...


class PeriodicEvent(AnyEvent):

    def __init__(self, name: str, period=0., start=0., mask: Func=None, hide_mask=False):...

    @property
    def period(self)->float:...

    @property
    def start(self)->float:...


class StopEvent(AnyEvent):

    def __init__(self, name: str, when: ObjFunc, check_if: BoolFunc=None, mask: Func=None, hide_mask=False, kill=False):...

    @property
    def when(self)->ObjFunc:...

    @property
    def check_if(self)->BoolFunc:...


class OdeResult:

    @property
    def t(self)->np.ndarray:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event_map(self)->dict[str, np.ndarray]:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->bool:...

    @property
    def success(self)->bool:...

    @property
    def runtime(self)->float:...

    @property
    def message(self)->str:...

    def examine(self):...

    def event_data(self, event: str)->tuple[np.ndarray, np.ndarray]:...


class SolverState:

    @property
    def t(self)->float:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event(self)->str:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->bool:...

    @property
    def is_running(self)->bool:...

    @property
    def is_dead(self)->bool:...

    @property
    def N(self)->int:...

    @property
    def message(self)->str:...

    def show(self):...


class ODE:

    def integrate(self, interval, *, max_frames=-1, max_events=-1, terminate=True, max_prints=0, include_first=False)->OdeResult:...

    def go_to(self, t, *, max_frames=-1, max_events=-1, terminate=True, max_prints=0, include_first=False)->OdeResult:...

    def advance(self)->bool:...

    def resume(self)->bool:...

    def free(self)->bool:...

    def state(self)->SolverState:...

    def copy(self)->ODE:...

    def save_data(self, savedir: str):...

    def clear(self):...

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
    def solver_filename(self)->str:...

    @property
    def runtime(self)->float:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->float:...

    @property
    def is_dead(self)->float:...


class LowLevelODE(ODE):

    def __init__(self, f, t0, q0, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45", events: list[AnyEvent]=None, mask=None, savedir="", save_events_only=False):...


class _DynamicODE(LowLevelODE):
    pass



def integrate_all(ode_array: Iterable[_DynamicODE], interval, max_frames=-1, max_events=-1, terminate=True, threads=-1, max_prints=0):...
