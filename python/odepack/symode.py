from __future__ import annotations
from typing import Iterable
from .odesolvers import * #type: ignore
from numiphy.lowlevelsupport import *


class SymbolicEvent:
    
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
        '''
        override
        '''
        return {"name": self.name, "hide_mask": self.hide_mask}
    
    def toEvent(self, **kwargs)->Event:...
    
    def funcs(self, t, *q, args)->dict[str, LowLevelCallable]:
        '''
        override
        '''
        raise NotImplementedError('')
    


class SymbolicPreciseEvent(SymbolicEvent):

    def __init__(self, name: str, event: Expr, dir=0, mask: Iterable[Expr]=None, hide_mask=False, event_tol=1e-12):
        SymbolicEvent.__init__(self, name, mask, hide_mask)
        self.event = event
        self.dir = dir
        self.event_tol = event_tol
    
    def __eq__(self, other):
        if type(other) is type(self):
            if other is self:
                return True
            else:
                return (self.event, self.dir, self.event_tol) == (other.event, other.dir, other.event_tol) and SymbolicEvent.__eq__(self, other)
        return False
    
    @property
    def kwargs(self):
        return dict(**SymbolicEvent.kwargs.__get__(self), dir=self.dir, event_tol=self.event_tol)
    
    def toEvent(self, when: ObjFunc, mask: Func=None, **__extra):
        return PreciseEvent(when=when, mask=mask, **self.kwargs, **__extra)
    
    def funcs(self, t, *q, args)->dict[str, LowLevelCallable]:
        ev_code = ScalarLowLevelCallable(self.event, t, q=q, args=args)

        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args)
        else:
            mask = None
        return {"when": ev_code, "mask": mask}


class SymbolicPeriodicEvent(SymbolicEvent):

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

    def toEvent(self, mask: Func=None, **__extra):
        '''
        override
        '''
        return PeriodicEvent(**self.kwargs, mask=mask, **__extra)

    def funcs(self, t, *q, args)->dict[str, LowLevelCallable]:
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args)
        else:
            mask = None
        return {"mask": mask}



class OdeSystem(CompileTemplate):

    _cls_instances: list[OdeSystem] = []

    _counter = 0

    ode_sys: tuple[Expr, ...]
    args: tuple[Symbol, ...]
    t: Symbol
    q: tuple[Symbol, ...]
    events: tuple[SymbolicEvent, ...]

    def __init__(self, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=(), module_name: str=None, directory: str = None):
        CompileTemplate.__init__(self, module_name=module_name, directory=directory)
        self._process_args(ode_sys, t, q, args=args, events=events)
    
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
        
        self.ode_sys = ode_sys
        self.args = args
        self.t = t
        self.q = q
        self.events = events
        for i in range(len(self.__class__._cls_instances)):
            if self.__class__._cls_instances[i] == self:
                return self.__class__._cls_instances[i]
        self.__class__._cls_instances.append(self)

    def __eq__(self, other: OdeSystem):
        if other is self:
            return True
        else:
            return (self.ode_sys, self.args, self.t, self.q, self.events) == (other.ode_sys, other.args, other.t, other.q, other.events)

    @property
    def Nsys(self):
        return len(self.ode_sys)
    
    @property
    def Nargs(self):
        return len(self.args)
    
    @cached_property
    def jacmat(self):
        array, symbols = self.ode_sys, self.q
        matrix = [[None for i in range(self.Nsys)] for j in range(self.Nsys)]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    @cached_property
    def ode_to_compile(self):
        return TensorLowLevelCallable(self.ode_sys, self.t, q=self.q, args=self.args)
    
    @property
    def jacobian_to_compile(self):
        return TensorLowLevelCallable(self.jacmat, self.t, q=self.q, args=self.args)
    
    @cached_property
    def _event_data(self)->tuple[dict[str, LowLevelCallable], ...]:
        res = ()
        for ev in self.events:
            res += (ev.funcs(self.t, *self.q, args=self.args),)
        return res
    
    @cached_property
    def _event_obj_map(self):
        return {event.name: event for event in self.events}
    
    @cached_property
    def true_events(self)->tuple[Event, ...]:
        res = []
        ptrs = self.pointers
        i=0
        for event, event_dict in zip(self.events, self._event_data):
            extra_kwargs = {"__Nsys": self.Nsys, "__Nargs": self.Nargs}
            for param_name, lowlevel_callable in event_dict.items():
                extra_kwargs[param_name] = ptrs[i+2]
                i+=1
            res.append(event.toEvent(**extra_kwargs))
        return res
    
    @cached_property
    def lowlevel_callables(self)->tuple[LowLevelCallable, ...]:
        event_objs: list[LowLevelCallable] = []
        for event_dict in self._event_data:
            for param_name, lowlevel_callable in event_dict.items():
                event_objs.append(lowlevel_callable)
        return (self.ode_to_compile, self.jacobian_to_compile, *event_objs)
    
    def get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->LowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return LowLevelODE(self.lowlevel_odefunc, t0=t0, q0=q0, jac=self.lowlevel_jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
    
    def get_variational(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->VariationalLowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return VariationalLowLevelODE(self.lowlevel_odefunc, t0=t0, q0=q0, jac=self.lowlevel_jac, period=period, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
    
    @cached_property
    def lowlevel_odefunc(self):
        return LowLevelFunction(self.pointers[0], self.Nsys, [self.Nsys], self.Nargs)
    
    @cached_property
    def lowlevel_jac(self):
        return LowLevelFunction(self.pointers[1], self.Nsys, [self.Nsys, self.Nsys], self.Nargs)
    
    @cached_property
    def _odefunc(self):
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.ode_sys, "numpy", self.t, q=self.q, **kwargs)
    
    @cached_property
    def _jac(self):
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.jacmat, "numpy", self.t, q=self.q, **kwargs)
    
    def odefunc(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        return self._odefunc(t, q, *args)
    
    def jac(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        return self._jac(t, q, *args)




def VariationalOdeSystem(ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], delq: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
    n = len(ode_sys)
    ode_sys = tuple(ode_sys)
    var_odesys = []
    for i in range(n):
        var_odesys.append(sum([ode_sys[i].diff(q[j])*delq[j] for j in range(n)]))
    
    new_sys = ode_sys + tuple(var_odesys)
    return OdeSystem(new_sys, t, [*q, *delq], args=args, events=events)


def load_ode_data(filedir: str)->tuple[np.ndarray[int], np.ndarray, np.ndarray]:
    data = np.loadtxt(filedir)
    events = data[:, 0].astype(int)
    t = data[:, 1].copy()
    q = data[:, 2:].copy()
    return events, t, q


def HamiltonianSystem2D(V: Expr, t: Symbol, x, y, px, py, args = (), events=()):
    q = [x, y, px, py]
    f = [px, py] + [-V.diff(x), -V.diff(y)]
    return OdeSystem(f, t, q, args=args, events=events)

def HamiltonianVariationalSystem2D(V: Expr, t: Symbol, x, y, px, py, delx, dely, delpx, delpy, args = (), events=()):
    q = [x, y, px, py]
    f = [px, py] + [-V.diff(x), -V.diff(y)]
    delq = [delx, dely, delpx, delpy]

    n = 4
    var_odesys = []
    for i in range(n):
        var_odesys.append(sum([f[i].diff(q[j])*delq[j] for j in range(n)]))
    
    new_sys = f + var_odesys

    return OdeSystem(new_sys, t, [*q, *delq], args=args, events=events)