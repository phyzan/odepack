import numpy as np
from typing import overload, Iterable
from .oderesult import OdeResult
from .ode import LowLevelODE
from .interp import *


class SampledScalarField:

    def __call__(self, *args: float)->float:
        '''
        Get the scalar field value at a specific point using interpolation.

        Parameters
        ----------
        *args : float
            Coordinates of the point at which to evaluate the scalar field. The number of arguments should match the dimensionality of the grid.

        Returns
        -------
        float
            Interpolated scalar field value at the given coordinates.
        '''
        ...


class SampledVectorField:

    def __init__(self, values: Iterable[np.ndarray], *grid: np.ndarray):
        '''
        Initialize a ND vector field on a regular grid.

        Parameters
        ----------
        values : Iterable of np.ndarray
            Vector field components at the grid points. The number of arrays should match the number of dimensions, and each array should have the same shape corresponding to the grid dimensions.
            For example, for a 3D vector field, values should be an iterable containing three arrays: (vx, vy, vz), where each array has shape (nx, ny, nz) corresponding to the grid dimensions.
        *grid : np.ndarray
            Coordinates of the grid points for each dimension. The number of grid arrays should match the number of dimensions, and the length of each grid array should match the corresponding dimension of the values arrays.

        Notes
        -----
        e.g. values[i][j, k] corresponds to grid point (x[j], y[k])
        '''
        ...


    def __call__(self, *args: float)->np.ndarray:
        '''
        Get the vector field components at a specific point using interpolation.

        Parameters
        ----------
        *args : float
            Coordinates of the point at which to evaluate the vector field. The number of arguments should match the dimensionality of the grid.

        Returns
        -------
        np.ndarray
            Array containing the vector field components at the given coordinates.
        '''
        ...


    def streamline(self, q0: np.ndarray, length: float, rtol = 1e-6, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, t_eval = None, method: str = "RK45") -> OdeResult:
        '''
        Compute a streamline starting from (x0, y0).

        Parameters
        ----------
        q0 : np.ndarray
            Initial coordinates
        length : float
            Length of the streamline to compute.
        rtol, atol : float, optional
            Relative and absolute tolerances for the ODE solver.
        min_step, max_step, stepsize : float, optional
            Step size control parameters.
        direction : {-1, 1}, optional
            Direction of integration. 1 for forward, -1 for backward.
        t_eval : array-like, optional
            Times at which to store solution values.
        method : str, optional
            Integration method. Default is "RK45".


        Returns
        -------
        OdeResult
            Result containing the computed streamline points.
        '''
        ...

    def get_ode(self, q0: np.ndarray, rtol = 1e-6, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, method: str = "RK45", normalized = False) -> LowLevelODE:
        '''
        Create a LowLevelODE object for streamlining from (x0, y0).

        Parameters
        ----------
        q0 : np.ndarray
            Initial coordinates.
        rtol, atol : float, optional
            Relative and absolute tolerances for the ODE solver.
        min_step, max_step, stepsize : float, optional
            Step size control parameters.
        direction : {-1, 1}, optional
            Direction of integration. 1 for forward, -1 for backward.
        method : str, optional
            Integration method. Default is "RK45".
        normalized : bool, optional
            If True, the vector field is normalized to unit length at each point, using the magnitude = sqrt(vx**2 + vy**2 + ...).
                Then, the integration parameter corresponds to arc length along the streamline.
        '''
        ...


class RegularGridScalarField(SampledScalarField, RegularGridInterpolator):

    def __init__(self, values: np.ndarray, *grid: np.ndarray):
        '''
        Initialize a scalar field on a regular grid.

        Parameters
        ----------
        values : np.ndarray
            Scalar field values at the grid points. Shape should match the dimensions of the grid.
        *grid : np.ndarray
            Coordinates of the grid points for each dimension. The number of grid arrays should match the number of dimensions, and the length of each grid array should match the corresponding dimension of the values array.

        Notes
        -----
        e.g. values[j, k] corresponds to grid point (x[j], y[k])
        '''
        ...


class RegularGridVectorField(SampledVectorField, RegularGridInterpolator):

    def __init__(self, values: Iterable[np.ndarray], *grid: np.ndarray):
        '''
        Initialize a vector field on a regular grid.

        Parameters
        ----------
        values : Iterable of np.ndarray
            Vector field components at the grid points. The number of arrays should match the number of dimensions, and each array should have the same shape corresponding to the grid dimensions.
            For example, for a 3D vector field, values should be an iterable containing three arrays: (vx, vy, vz), where each array has shape (nx, ny, nz) corresponding to the grid dimensions.
        *grid : np.ndarray
            Coordinates of the grid points for each dimension. The number of grid arrays should match the number of dimensions, and the length of each grid array should match the corresponding dimension of the values arrays.

        Notes
        -----
        e.g. values[i][j, k] corresponds to grid point (x[j], y[k])
        '''
        ...

    def component(self, i: int)->np.ndarray:
        '''
        Get the i-th component of the vector field as a scalar field on the same grid.

        Parameters
        ----------
        i : int
            Index of the component to retrieve. Should be in the range [0, n_dimensions-1].

        Returns
        -------
        np.ndarray
            Scalar field values corresponding to the i-th component of the vector field, with the same shape as the grid.
        '''
        ...

    def streamplot_data(self, max_length: float, ds: float, density: int = 30)->list[np.ndarray]:
        '''
        Compute streamplot data for visualization.

        Parameters
        ----------
        max_length : float
            Maximum length of streamlines to compute.
        ds: float
            Step size for the RK4 integrator.
        density : int, optional
            Density of streamlines. The number of streamlines per axis will be approximately equal to the density. Default is 30.
        
        Returns
        -------
        list of np.ndarray
            List of arrays containing streamline points for visualization.
            x_line, y_line = result[i]
        '''
        ...


class ScatteredScalarField(SampledScalarField, ScatteredNdInterpolator):

    @overload
    def __init__(self, points: np.ndarray, values: np.ndarray):
        '''
        Initialize a scattered scalar field, for interpolation from scattered data points in any number of dimensions.
        Linear interpolation is used.


        Parameters
        ----------
        points : np.ndarray
            Coordinates of the scattered points. Shape should be (n_points, n_dimensions).
        values : np.ndarray
            Scalar field values at the scattered points. Shape should be (n_points,).
        '''
        ...

    @overload
    def __init__(self, tri: DelaunayTri, values: np.ndarray):
        '''
        Initialize a scattered scalar field from a Delaunay triangulation and corresponding scalar values.

        Parameters
        ----------
        tri : DelaunayTri
            Delaunay triangulation of the scattered points.
        values : np.ndarray
            Scalar field values at the scattered points. Shape should be (n_points,).
        '''
        ...

    @property
    def points(self)->np.ndarray:
        '''
        Coordinates of the scattered points.
        '''
        ...

    @property
    def values(self)->np.ndarray:
        '''
        Scalar field values at the scattered points.
        '''
        ...

    @property
    def tri(self)->DelaunayTri:
        '''
        Delaunay triangulation of the scattered points.
        '''
        ...
        
class ScatteredVectorField(SampledVectorField, ScatteredNdInterpolator):

    @overload
    def __init__(self, points: np.ndarray, values: Iterable[np.ndarray]):
        '''
        Initialize a scattered vector field, for interpolation from scattered data points in any number of dimensions.
        Linear interpolation is used.

        Parameters
        ----------
        points : np.ndarray
            Coordinates of the scattered points. Shape should be (n_points, n_dimensions).
        values : Iterable of np.ndarray
            Vector field components at the scattered points. The number of arrays should match the number of dimensions, and each array should have shape (n_points,).
            For example, for a 3D vector field, values should be an iterable containing three arrays: (vx, vy, vz), where each array has shape (n_points,) corresponding to the scattered points.
        '''
        ...

    @overload
    def __init__(self, tri: DelaunayTri, values: Iterable[np.ndarray]):
        '''
        Initialize a scattered vector field from a Delaunay triangulation and corresponding vector values.

        Parameters
        ----------
        tri : DelaunayTri
            Delaunay triangulation of the scattered points.
        values : Iterable of np.ndarray
            Vector field components at the scattered points. The number of arrays should match the number of dimensions, and each array should have shape (n_points,).
            For example, for a 3D vector field, values should be an iterable containing three arrays: (vx, vy, vz), where each array has shape (n_points,) corresponding to the scattered points.
        '''
        ...