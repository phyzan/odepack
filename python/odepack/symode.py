from __future__ import annotations
from typing import Iterable
from .odesolvers_double import * #type: ignore
from .odesolvers_float import * #type: ignore
from .odesolvers_longdouble import * #type: ignore
from .odesolvers_mpreal import * #type: ignore
# from . import LowLevelODE_Double, LowLevelODE_Float, LowLevelODE_LongDouble, LowLevelODE_MpReal, VariationalLowLevelODE_Double, VariationalLowLevelODE_Float, VariationalLowLevelODE_LongDouble, VariationalLowLevelODE_MpReal, PreciseEvent_Double, PreciseEvent_Float, PreciseEvent_LongDouble, PreciseEvent_MpReal, PeriodicEvent_Double, PeriodicEvent_Float, PeriodicEvent_LongDouble, PeriodicEvent_MpReal
from numiphy.lowlevelsupport import *
from typing import TypeVar
T = TypeVar('T')


def lowlevel_ode(f, t0: float, q0: np.ndarray, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), events=(), method="RK45", dtype='double'):
    if dtype == 'double':
        return LowLevelODE_Double(f=f, t0=t0, q0=q0, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'float':
        return LowLevelODE_Float(f=f, t0=t0, q0=q0, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'long double':
        return LowLevelODE_LongDouble(f=f, t0=t0, q0=q0, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'mpreal':
        return LowLevelODE_MpReal(f=f, t0=t0, q0=q0, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float', 'double', 'long double', and 'mpreal'.")

def precise_event(name: str, when, dir=0, mask=None, hide_mask=False, event_tol=1e-12, dtype='double', **__extra):
    if dtype == 'double':
        return PreciseEvent_Double(name=name, when=when, dir=dir, mask=mask, hide_mask=hide_mask, event_tol=event_tol, **__extra)
    elif dtype == 'float':
        return PresiceEvent_Float(name=name, when=when, dir=dir, mask=mask, hide_mask=hide_mask, event_tol=event_tol, **__extra)
    elif dtype == 'long double':
        return PresiceEvent_LongDouble(name=name, when=when, dir=dir, mask=mask, hide_mask=hide_mask, event_tol=event_tol, **__extra)
    elif dtype == 'mpreal':
        return PresiceEvent_MpReal(name=name, when=when, dir=dir, mask=mask, hide_mask=hide_mask, event_tol=event_tol, **__extra)
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float', 'double', 'long double', and 'mpreal'.")
    
def periodic_event(name: str, period: float, start = None, mask=None, hide_mask=False, dtype='double', **__extra):
    if dtype == 'double':
        return PeriodicEvent_Double(name=name, period=period, start=start, mask=mask, hide_mask=hide_mask, **__extra)
    elif dtype == 'float':
        return PeriodicEvent_Float(name=name, period=period, start=start, mask=mask, hide_mask=hide_mask, **__extra)
    elif dtype == 'long double':
        return PeriodicEvent_LongDouble(name=name, period=period, start=start, mask=mask, hide_mask=hide_mask, **__extra)
    elif dtype == 'mpreal':
        return PeriodicEvent_MpReal(name=name, period=period, start=start, mask=mask, hide_mask=hide_mask, **__extra)
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float', 'double', 'long double', and 'mpreal'.")

def variational_lowlevel_ode(f, t0: float, q0: np.ndarray, period: float, *, jac: Callable = None, rtol=1e-12, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), events=(), method="RK45", dtype='double'):
    if dtype == 'double':
        return VariationalLowLevelODE_Double(f=f, t0=t0, q0=q0, period=period, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'float':
        return VariationalLowLevelODE_Float(f=f, t0=t0, q0=q0, period=period, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'long double':
        return VariationalLowLevelODE_LongDouble(f=f, t0=t0, q0=q0, period=period, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    elif dtype == 'mpreal':
        return VariationalLowLevelODE_MpReal(f=f, t0=t0, q0=q0, period=period, jac=jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float', 'double', 'long double', and 'mpreal'.")


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
    
    def toEvent(self, dtype, **kwargs):...
    
    def funcs(self, t, *q, args, dtype='double')->dict[str, LowLevelCallable]:
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
    
    def toEvent(self, dtype, when, mask=None, **__extra):
        return precise_event(dtype=dtype, when=when, mask=mask, **self.kwargs, **__extra)
    
    def funcs(self, t, *q, args, dtype='double')->dict[str, LowLevelCallable]:
        ev_code = ScalarLowLevelCallable(self.event, t, q=q, args=args, scalar_type=dtype)

        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args, scalar_type=dtype)
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

    def toEvent(self, dtype, mask=None, **__extra):
        '''
        override
        '''
        return periodic_event(dtype=dtype, **self.kwargs, mask=mask, **__extra)

    def funcs(self, t, *q, args, dtype='double')->dict[str, LowLevelCallable]:
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args, scalar_type=dtype)
        else:
            mask = None
        return {"mask": mask}



class OdeSystem:

    _cls_instances: list[OdeSystem] = []

    _counter = 0

    ode_sys: tuple[Expr, ...]
    args: tuple[Symbol, ...]
    t: Symbol
    q: tuple[Symbol, ...]
    events: tuple[SymbolicEvent, ...]

    def __init__(self, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=(), module_name: str=None, directory: str = None):
        self.__directory = directory if directory is not None else tools.get_source_dir()
        self.__module_name = module_name if module_name is not None else tools.random_module_name()
        self.__nan_dir = directory is None
        self.__nan_modname = module_name is None
        self._process_args(ode_sys, t, q, args=args, events=events)
        # Initialize caches for dtype-specific low-level functions
        self._pointers_cache = {}
    
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
    def directory(self):
        return self.__directory
    
    def module_name(self, dtype='double'):
        return f'{self.__module_name}_{dtype.replace(" ", "_")}'
    
    @property
    def Nsys(self):
        return len(self.ode_sys)
    
    @property
    def Nargs(self):
        return len(self.args)
    
    @cached_property
    def python_funcs(self)->tuple[Callable,...]:
        llc = self.lowlevel_callables()
        res = [None for _ in llc]
        for i in range(len(res)):
            if llc[i] is not None:
                f = llc[i].to_python_callable().lambdify()
                res[i] = lambda t, q, *args : f(t, q, args)
        return tuple(res)
    
    @cached_property
    def jacmat(self):
        array, symbols = self.ode_sys, self.q
        matrix = [[None for i in range(self.Nsys)] for j in range(self.Nsys)]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    def code(self, dtype='double')->str:
        return generate_cpp_code(self.lowlevel_callables(dtype=dtype), self.module_name(dtype=dtype))
    
    def compile(self, dtype='double')->tuple:
        if not self.__nan_dir:
            with open(self._cpp_path(dtype=dtype), "w") as f:
                f.write(self.code(dtype=dtype))
        result = compile_funcs(self.lowlevel_callables(dtype=dtype), self.directory, self.module_name(dtype=dtype))
        return result
    
    
    def ode_to_compile(self, dtype='double')->TensorLowLevelCallable:
        return TensorLowLevelCallable(self.ode_sys, self.t, q=self.q, scalar_type=dtype, args=self.args)
    
    
    def jacobian_to_compile(self, dtype='double')->TensorLowLevelCallable:
        return TensorLowLevelCallable(self.jacmat, self.t, q=self.q, scalar_type=dtype, args=self.args)
    
    
    def _event_data(self, dtype='double')->tuple[dict[str, LowLevelCallable], ...]:
        res = ()
        for ev in self.events:
            res += (ev.funcs(self.t, *self.q, args=self.args, dtype=dtype),)
        return res
    
    @cached_property
    def _event_obj_map(self):
        return {event.name: event for event in self.events}
    
    def _cpp_path(self, dtype='double'):
        return os.path.join(self.directory, f"ode_callables_{dtype.replace(' ', '_')}.cpp")
    
    def _pointers(self, dtype='double')->tuple[int, ...]:
        if dtype in self._pointers_cache:
            return self._pointers_cache[dtype]
        path = self._cpp_path(dtype=dtype)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    saved_code = f.read()
                if saved_code == self.code(dtype=dtype):
                    return tools.import_lowlevel_module(self.directory, self.module_name(dtype=dtype)).pointers()
            except:
                pass
        res = self.compile(dtype=dtype)
        self._pointers_cache[dtype] = res
        return res
    
    def _true_events(self, compiled=True, dtype='double'):
        res = []
        if compiled:
            items = self._pointers(dtype=dtype)
        else:
            items = self.python_funcs
        i=0
        for event, event_dict in zip(self.events, self._event_data(dtype=dtype)):
            extra_kwargs = {"__Nsys": self.Nsys, "__Nargs": self.Nargs}
            for param_name, lowlevel_callable in event_dict.items():
                extra_kwargs[param_name] = items[i+2]
                i+=1
            res.append(event.toEvent(dtype=dtype, **extra_kwargs))
        return res

    def true_compiled_events(self, dtype='double'):
        return self._true_events(compiled=True, dtype=dtype)
    
    @cached_property
    def true_py_events(self):
        return self._true_events(compiled=False)
    
    def lowlevel_callables(self, dtype='double')->tuple[LowLevelCallable, ...]:
        event_objs: list[LowLevelCallable] = []
        for event_dict in self._event_data(dtype=dtype):
            for param_name, lowlevel_callable in event_dict.items():
                event_objs.append(lowlevel_callable)
        return (self.ode_to_compile(dtype=dtype), self.jacobian_to_compile(dtype=dtype), *event_objs)
    
    def _get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", period=None, compiled=True, dtype='double'):
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        if compiled:
            pointers = self._pointers(dtype=dtype)
            events = self.true_compiled_events(dtype=dtype)
            f, jac = pointers[:2]
        else:
            events = self.true_py_events
            f = self._odefunc
            jac = self._jac
        kwargs = dict(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, events=events, method=method)
        if period is None:
            getter = lowlevel_ode
        else:
            kwargs['period'] = period
            getter = variational_lowlevel_ode
        return getter(f=f, jac=jac, **kwargs)
    
    def get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", compiled=True, dtype='double'):
        return self._get(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, method=method, period=None, compiled=compiled, dtype=dtype)
    
    def get_variational(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., direction=1, args=(), method="RK45", compiled=True, dtype='double'):
        return self._get(t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, direction=direction, args=args, method=method, period=period, compiled=compiled, dtype=dtype)
    
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