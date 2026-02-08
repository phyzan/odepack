from __future__ import annotations

from numiphy.symlib import Expr, Symbol, asexpr, S, InterpedArray
from numiphy.symlib.hashing import _HashableNdArray, _HashableGrid
from numiphy.findiffs import Grid
from functools import cached_property
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from .odesolvers import * #type: ignore


class RegularScalarField(Expr):

    _priority = 34 # TODO: choose correct

    def __new__(cls, name: str, grid: Grid, values: np.ndarray, *expressions: Expr, simplify=True):
        '''
        A regular scalar field defined on a grid, with values at the grid points and optional expressions for interpolation.

        Parameters
        ----------
        name : str
            The name of the scalar field.
        grid : Grid
            The grid on which the scalar field is defined.
        values : np.ndarray
            The values of the scalar field at the grid points. Must have the same shape as the grid.
        *expressions : Expr
            Optional expressions for interpolation on each axis. The number of expressions must match the number of dimensions of the grid.
        '''

        if grid.shape != values.shape:
            raise ValueError(f"Grid shape {grid.shape} does not match values shape {values.shape}")
        elif grid.nd != len(expressions):
            raise ValueError(f"Number of expressions {len(expressions)} does not match grid dimensions {grid.nd}")
        return super().__new__(cls, name, grid, values, *expressions)
    
    @property
    def name(self) -> str:
        return self.args[0]
    
    @property
    def grid(self) -> Grid:
        return self.args[1]
    
    @property
    def values(self) -> np.ndarray:
        return self.args[2]
    
    @property
    def expressions(self) -> tuple[Expr, ...]:
        return self.args[3:]
    
    @property
    def ndim(self) -> int:
        return self.grid.nd
    
    @cached_property
    def to_sampled_scalar_field(self):
        '''Create a precompiled-class object'''
        if self.ndim == 1:
            cls = SampledScalarField1D
        elif self.ndim == 2:
            cls = SampledScalarField2D
        elif self.ndim == 3:
            cls = SampledScalarField3D
        else:
            raise NotImplementedError(f"Conversion to sampled scalar field not implemented for dimension {self.ndim}")
        return cls(self.values, *self.grid.x)

    
    def __call__(self, *args: Expr) -> RegularScalarField:
        '''
        Substitute new expressions for interpolation, regardless of the current expressions. The number of new expressions must match the number of dimensions of the grid.
        '''
        if len(args) != self.grid.nd:
            raise ValueError(f"Number of new expressions {len(args)} does not match grid dimensions {self.grid.nd}")
        return RegularScalarField(self.name, self.grid, self.values, *args)
    
    def repr(self, lib="", **kwargs):
        return f'{self.name}({", ".join([x.repr(lib=lib, **kwargs) for x in self.expressions])})'
    

    def __eq__(self, other):
        if not isinstance(other, RegularScalarField):
            return False
        elif self is other:
            return True
        elif (self.values is other.values):
            return self.grid == other.grid and self.expressions == other.expressions
        elif self.values.shape == other.values.shape:
            return np.all(self.values == other.values) and self.args[1:] == other.args[1:]
        else:
            return False
        
    def __hash__(self):
        return hash((self.__class__,) + self._hashable_content)
    
    @cached_property
    def interpolator(self):
        return RegularGridInterpolator(self.grid.x, self.values, method="cubic")
    
    @property
    def _hashable_content(self):
        return (_HashableNdArray(self.values), _HashableGrid(self.grid), *self.expressions)

    def _diff(self, var):
        '''
        d/dx P (f1(x, y), f2(x, y), ... ) = sum_i dP/dfi * dfi/dx
        '''
        res = S.Zero
        P = InterpedArray(self.values, self.grid) # just a scalar field that can diff wrt each axis using finite differences

        for axis, fi in enumerate(self.expressions):
            dP_dfi = P.diff(axis)
            dfi_dvar = fi.diff(var)
            res += RegularScalarField(f'{self.name}_{axis}', dP_dfi.grid, dP_dfi.ndarray(), *self.expressions) * dfi_dvar
        return res
    
    def eval(self):
        if len(self.symbols) == 0:
            return self.interpolator(tuple([expr.eval().value for expr in self.expressions]))
        else:
            return Expr.eval(self)
        
    def lowlevel_repr(self, scalar_type="double"):
        return f'(*{self.name})({", ".join([x.lowlevel_repr(scalar_type) for x in self.expressions])})'