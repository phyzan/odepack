from __future__ import annotations

from matplotlib.pyplot import grid
from numiphy.symlib import Expr, Symbol, asexpr, S, InterpedArray
from numiphy.symlib.hashing import _HashableNdArray, _HashableGrid
from numiphy.findiffs import Grid
from functools import cached_property
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from .odesolvers import * #type: ignore


class NumericalScalarField(Expr):

    @property
    def name(self) -> str:
        raise NotImplementedError('Must be implemented by subclasses')

    @property
    def c_class_name(self):
        raise NotImplementedError('Must be implemented by subclasses')

    @property
    def compiled_signature(self):
        return f'const ode::{self.c_class_name}* {self.name} = nullptr;'
    
    @property
    def _compiled_obj(self):
        raise NotImplementedError('Must be implemented by subclasses')
    
    @cached_property
    def to_sampled_scalar_field(self):
        '''
        Returns self._compiled_obj. It is important that this is a cached property, so that
        when this class is used in the OdeSystem class, since this NumericalScalarField object will be cached in the OdeSystem class itself (not just a current instance), the memory address of the compiled object will be valid for the lifetime of extracted compiled ODE objects.
        '''
        return self._compiled_obj

class RegularScalarField(NumericalScalarField):

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
    def c_class_name(self):
        return f'PyScalarField<{self.ndim}>'
    
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
    
    @property
    def _compiled_obj(self):
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
            return np.all(self.values == other.values) and self.args[1] == other.args[1] and self.args[3:] == other.args[3:]
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
    
    def diff(self, var, order=1) -> RegularScalarField:
        return super().diff(var, order)
    
    def as_unstructured(self, new_name=None):
        '''
        Convert this regular scalar field to an irregular scalar field with the same values and expressions, but with the grid points flattened into a single array of coordinates. The new grid points will be the same as the original grid points, but flattened into a single array of coordinates. The new values will be the same as the original values, but flattened into a single array. The new expressions will be the same as the original expressions.

        Parameters
        ----------
        new_name : str
            The name of the new irregular scalar field. If None, the name will be the same as the original name with "_unstructured" appended.
        '''
        X, Y = np.meshgrid(*self.grid.x, indexing="ij")
        coords = np.stack([X, Y], axis=-1).reshape(-1, self.ndim)
        values = self.values.reshape(-1) # shape (npoints,)
        name = new_name if new_name is not None else f'{self.name}_unstructured'
        return IrregularScalarField(name, coords, values, *self.expressions)
    


class IrregularScalarField(NumericalScalarField):

    _priority = 35 # TODO: choose correct

    def __new__(cls, name: str, points: np.ndarray | DelaunayTriangulation, values: np.ndarray, *expressions: Expr, simplify=True):
        '''
        An irregular scalar field defined on scattered points, with values at those points and optional expressions for interpolation.

        Parameters
        ----------
        name : str
            The name of the scalar field.
        points: np.ndarray or any Delaunay class, as long as the dimensions match with the values and expressions
            The coordinates of the scattered points where the scalar field is defined. Must have shape (n_points, n_dimensions) or (n_points,).
        values : np.ndarray
            The values of the scalar field at the scattered points. Must have the same length as the number of points.
        *expressions : Expr
            Optional expressions for interpolation on each axis. The number of expressions must match the number of dimensions of the grid.

        IMPORTANT
        ----------------------
        The name will be th unique identifiers used for comparison with other instances of this class, because comparing the points and values is expensive
        when constructing a symbolic expression. Make sure to give different names to objects with different points or values.
        '''
        expressions = tuple([asexpr(x) for x in expressions])
        if isinstance(points, (DelaunayTriangulation)):
            points, tri = points.points, points
        else:
            tri = None

        if points.ndim not in (1, 2):
            raise ValueError("ScatteredField requires a 2D array for the input points (or optionally a 1D array for 1D fields)")
        elif values.ndim != 1:
            raise ValueError("ScatteredField requires a 1D array for the field values")
        elif points.shape[0] != values.size:
            raise ValueError("Number of input points must match number of field values")
        elif points.ndim == 2 and points.shape[0] < points.shape[1]+1:
            raise ValueError("Number of input points must be at least one more than the number of dimensions")
        elif points.ndim == 2 and len(expressions) != points.shape[1]:
            raise ValueError(f"Number of expressions {len(expressions)} must match number of dimensions {points.shape[1]}")
        elif points.ndim == 1 and len(expressions) != 1:
            raise ValueError(f"Number of expressions {len(expressions)} must be 1 for 1D fields")
        
        obj = super().__new__(cls, name, points, values, *expressions)
        if tri is not None:
            obj.__dict__['tri'] = tri # store triangulation if it was provided, so that it can be reused in interpolation without needing to recompute
        return obj
    
    @property
    def tri_is_cached(self):
        return 'tri' in self.__dict__
    
    @cached_property
    def tri(self)->DelaunayTriangulation:
        print(f"Computing Delaunay triangulation for irregular scalar field for {self.name}") # this can be expensive, so it's good to have a print statement to indicate when it's happening
        if self.ndim == 1:
            return DelaunayTri1D(self.points)
        elif self.ndim == 2:
            return DelaunayTri2D(self.points)
        elif self.ndim == 3:
            return DelaunayTri3D(self.points)
        else:
            return DelaunayTriND(self.points)
    
    @property
    def c_class_name(self):
        ndim = self.ndim if self.ndim <= 3 else '0'
        return f'PyScatteredField<{ndim}>'
    
    @property
    def name(self) -> str:
        return self.args[0]
    
    @property
    def points(self) -> np.ndarray:
        return self.args[1]
    
    @property
    def values(self) -> np.ndarray:
        return self.args[2]
    
    @property
    def expressions(self) -> tuple[Expr, ...]:
        return self.args[3:]
    
    @property
    def ndim(self) -> int:
        if self.points.ndim == 1:
            return 1
        else:
            return self.points.shape[1]
    
    @property
    def _compiled_obj(self):
        '''Create a precompiled-class object'''
        if self.ndim == 1:
            cls = ScatteredScalarField1D
        elif self.ndim == 2:
            cls = ScatteredScalarField2D
        elif self.ndim == 3:
            cls = ScatteredScalarField3D
        else:
            cls = ScatteredScalarFieldND
        
        return cls(self.tri, self.values)

    
    def __call__(self, *args: Expr) -> IrregularScalarField:
        '''
        Substitute new expressions for interpolation, regardless of the current expressions. The number of new expressions must match the number of dimensions of the grid.
        '''
        if len(args) != self.ndim:
            raise ValueError(f"Number of new expressions {len(args)} does not match grid dimensions {self.ndim}")
        new_field = IrregularScalarField(self.name, self.points, self.values, *args)
        # Propagate cached triangulation if it exists
        if 'tri' in self.__dict__:
            new_field.__dict__['tri'] = self.tri
        return new_field
    
    def repr(self, lib="", **kwargs):
        return f'{self.name}({", ".join([x.repr(lib=lib, **kwargs) for x in self.expressions])})'
    

    def __eq__(self, other):
        if not isinstance(other, IrregularScalarField):
            return False
        else:
            return self.name == other.name # only compare names, since comparing points and values is expensive. The name should be unique for different points and values, so this should be sufficient for correct behavior in symbolic expressions.
        
    def __hash__(self):
        return hash((self.__class__,) + self._hashable_content)
    
    @property
    def _hashable_content(self):
        return (self.name, *self.expressions)
    
    def eval(self):
        if len(self.symbols) == 0:
            return self.to_sampled_scalar_field(*tuple([expr.eval().value for expr in self.expressions]))
        else:
            return Expr.eval(self)
        
    def lowlevel_repr(self, scalar_type="double"):
        return f'(*{self.name})({", ".join([x.lowlevel_repr(scalar_type) for x in self.expressions])})'