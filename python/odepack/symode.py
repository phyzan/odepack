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
    
    def init_code(self, var_name, t, *q, args)->dict[str, tuple[str, str]]:...
    


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
    
    def init_code(self, var_name, t, *q, args)->dict[str, tuple[str, str]]:
        names = [f'{var_name}_EVENT', f'{var_name}_CHECK', f'{var_name}_MASK']
        ev_code = ScalarLowLevelCallable(self.event, t, q=q, args=args).code(names[0])

        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args).code(names[2])
        else:
            mask = f"void* {names[2]} = nullptr;"
        return {"when": (names[0], ev_code), "mask": (names[2], mask)}


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

    def init_code(self, var_name, t, *q, args)->dict[str, tuple[str, str]]:
        name = f"{var_name}_MASK"
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args).code(name)
        else:
            mask = f"void* {name} = nullptr;"
        return {"mask": (name, mask)}



class OdeSystem:

    _cls_instances: list[OdeSystem] = []

    _counter = 0

    ode_sys: tuple[Expr, ...]
    args: tuple[Symbol, ...]
    t: Symbol
    q: tuple[Symbol, ...]
    events: tuple[SymbolicEvent, ...]

    def __new__(cls, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
        obj = object.__new__(cls)
        return cls._process_args(obj, ode_sys, t, q, args=args, events=events)
    
    @classmethod
    def _process_args(cls, obj: OdeSystem, ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
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
        
        obj.ode_sys = ode_sys
        obj.args = args
        obj.t = t
        obj.q = q
        obj.events = events
        for i in range(len(cls._cls_instances)):
            if cls._cls_instances[i] == obj:
                return cls._cls_instances[i]
        cls._cls_instances.append(obj)
        return obj

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
    
    @property
    def module_name(self):
        return f'ODE_MODULE_{self.__class__._counter}'
    
    @cached_property
    def jacmat(self):
        array, symbols = self.ode_sys, self.q
        matrix = [[None for i in range(self.Nsys)] for j in range(self.Nsys)]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    @property
    def odefunc_code(self):
        f = TensorLowLevelCallable(self.ode_sys, self.t, q=self.q, args=self.args)
        return f.code("ODE_FUNC")
    
    @property
    def jacobian_code(self):
        f = TensorLowLevelCallable(self.jacmat, self.t, q=self.q, args=self.args)
        return f.code("JAC_FUNC")
    
    @cached_property
    def _event_dict(self)->dict[str, dict[str, tuple[str, str]]]:
        events = {}
        for i, ev in enumerate(self.events):
            events[ev.name] = ev.init_code(f'EVENT_{i+1}', self.t, *self.q, args=self.args)
        return events
    
    @cached_property
    def _event_obj_map(self):
        return {event.name: event for event in self.events}
    
    @cached_property
    def _func_names(self)->list[str]:
        res = []
        for name, data in self._event_dict.items():
            for param_name, (func_name, func_code) in data.items():
                res.append(func_name)
        return ["ODE_FUNC", "JAC_FUNC", *res]
    
    @cached_property
    def true_events(self)->tuple[Event, ...]:
        res = []
        ptrs = self._pointers
        i=0
        for name, data in self._event_dict.items():
            extra_kwargs = {"__Nsys": self.Nsys, "__Nargs": self.Nargs}
            for param_name, (func_name, func_code) in data.items():
                extra_kwargs[param_name] = ptrs[i+2]
                i+=1
            res.append(self._event_obj_map[name].toEvent(**extra_kwargs))
        return res

    def event_block(self)->str:
        res = ''
        for name, data in self._event_dict.items():
            for param_name, (func_name, func_code) in data.items():
                res += func_code+'\n\n'
        return res

    def module_code(self, name = "ode_module"):
        header = "#include <pybind11/pybind11.h>\n\n#include <complex>\n\nusing std::complex, std::imag, std::real, std::numbers::pi;\n\nnamespace py = pybind11;"
        event_block = self.event_block()
        ode_func = self.odefunc_code
        jac_func = self.jacobian_code

        array = "py::make_tuple("+", ".join([f'reinterpret_cast<const void*>({fname})' for fname in self._func_names])+")"

        py_func = '\n\tm.def("pointers", [](){return '+array+';});'
        pybind_cond = f"PYBIND11_MODULE({name}, m)"+'{'+py_func+'\n}'
        items = [header, event_block, ode_func, jac_func, pybind_cond]
        return "\n\n".join(items)

    def generate_cpp_file(self, directory, module_name):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.module_code(name=module_name)
        cpp_file = os.path.join(directory, f"{module_name}.cpp")
        with open(cpp_file, "w") as f:
            f.write(code)
        
        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name: str):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name)
            tools.compile(cpp_file, directory, module_name)
    
    def get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->LowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return LowLevelODE(self.lowlevel_odefunc, t0=t0, q0=q0, jac=self.lowlevel_jac, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
    
    def get_variational(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->VariationalLowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return VariationalLowLevelODE(self.lowlevel_odefunc, t0=t0, q0=q0, jac=self.lowlevel_jac, period=period, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
    
    @cached_property
    def _pointers(self)->list:
        
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname)
            temp_module = tools.import_lowlevel_module(so_dir, modname)
        
        self.__class__._counter += 1
        return temp_module.pointers()
    
    @cached_property
    def lowlevel_odefunc(self):
        return LowLevelFunction(self._pointers[0], self.Nsys, [self.Nsys], self.Nargs)
    
    @cached_property
    def lowlevel_jac(self):
        return LowLevelFunction(self._pointers[1], self.Nsys, [self.Nsys, self.Nsys], self.Nargs)
    
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