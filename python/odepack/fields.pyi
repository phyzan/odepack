import numpy as np
from typing import overload
from .oderesult import OdeResult
from .ode import LowLevelODE


class SampleVectorField:


    def streamline(self, q0: np.ndarray, length: float, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, t_eval = None, method: str = "RK45") -> OdeResult:
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

    def get_ode(self, q0: np.ndarray, rtol = 1e-12, atol = 1e-12, min_step = 0., max_step = None, stepsize = 0., direction=1, method: str = "RK45", normalized = False) -> LowLevelODE:
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


class SampledVectorField2D:

    '''
    Class representing a 2D vector field
    '''

    def __init__(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray):
        '''
        Initialize a 2D vector field.
        Parameters
        ----------
        x : np.ndarray
            X-coordinates of the grid points.
        y : np.ndarray
            Y-coordinates of the grid points.
        vx : np.ndarray
            X-components of the vector field at the grid points.
        vy : np.ndarray
            Y-components of the vector field at the grid points.

        Notes
        -----
        The shape of vx and vy must be (len(x), len(y))
        and vx[i, j], vy[i, j] correspond to the vector at (x[i], y[j]).

        '''
        ...

    @property
    def x(self)->np.ndarray:
        '''
        X-coordinates of the grid points.
        '''
        ...

    @property
    def y(self)->np.ndarray:
        '''
        Y-coordinates of the grid points.
        '''
        ...

    @property
    def vx(self)->np.ndarray:
        '''
        X-components of the vector field at the grid points.
        '''
        ...

    @property
    def vy(self)->np.ndarray:
        '''
        Y-components of the vector field at the grid points.
        '''
        ...

    def __call__(self, x: float, y: float)->np.ndarray:
        '''
        Get the vector field components (vx, vy) at a specific point (x, y) using interpolation.
        This is equivalent to np.array([self.get_vx(x, y), self.get_vy(x, y)])
        '''
        ...

    def get_vx(self, x: float, y: float)->float:
        '''
        Get the x-component of the vector field at a specific point (x, y) using interpolation.
        '''
        ...

    def get_vy(self, x: float, y: float)->float:
        '''
        Get the y-component of the vector field at a specific point (x, y) using interpolation.
        '''
        ...

    def in_bounds(self, x: float, y: float)->bool:
        '''
        Check if the point (x, y) is within the bounds of the grid.
        '''
        ...


class SampledVectorField3D:

    '''
    Class representing a 3D vector field
    '''

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
        '''
        Initialize a 3D vector field.
        Parameters
        ----------
        x : np.ndarray
            X-coordinates of the grid points.
        y : np.ndarray
            Y-coordinates of the grid points.
        z : np.ndarray
            Z-coordinates of the grid points.
        vx : np.ndarray
            X-components of the vector field at the grid points.
        vy : np.ndarray
            Y-components of the vector field at the grid points.
        vz : np.ndarray
            Z-components of the vector field at the grid points.

        Notes
        -----
        The shape of vx, vy, and vz must be (len(x), len(y), len(z))
        and vx[i, j, k], vy[i, j, k], vz[i, j, k] correspond to the vector at (x[i], y[j], z[k]).

        '''
        ...

    @property
    def x(self)->np.ndarray:
        '''
        X-coordinates of the grid points.
        '''
        ...

    @property
    def y(self)->np.ndarray:
        '''
        Y-coordinates of the grid points.
        '''
        ...

    @property
    def z(self)->np.ndarray:
        '''
        Z-coordinates of the grid points.
        '''
        ...

    @property
    def vx(self)->np.ndarray:
        '''
        X-components of the vector field at the grid points.
        '''
        ...

    @property
    def vy(self)->np.ndarray:
        '''
        Y-components of the vector field at the grid points.
        '''
        ...

    @property
    def vz(self)->np.ndarray:
        '''
        Z-components of the vector field at the grid points.
        '''
        ...

    def __call__(self, x: float, y: float, z: float)->np.ndarray:
        '''
        Get the vector field components (vx, vy, vz) at a specific point (x, y, z) using interpolation.
        This is equivalent to np.array([self.get_vx(x, y, z), self.get_vy(x, y, z), self.get_vz(x, y, z)])
        '''
        ...

    def get_vx(self, x: float, y: float, z: float)->float:
        '''
        Get the x-component of the vector field at a specific point (x, y, z) using interpolation.
        '''
        ...

    def get_vy(self, x: float, y: float, z: float)->float:
        '''
        Get the y-component of the vector field at a specific point (x, y, z) using interpolation.
        '''
        ...

    def get_vz(self, x: float, y: float, z: float)->float:
        '''
        Get the z-component of the vector field at a specific point (x, y, z) using interpolation.
        '''
        ...

    def in_bounds(self, x: float, y: float, z: float)->bool:
        '''
        Check if the point (x, y, z) is within the bounds of the grid.
        '''
        ...


class SampledScalarField1D:

    def __init__(self, f: np.ndarray, x: np.ndarray):
        '''
        Initialize a 1D scalar field.
        Parameters
        ----------
        f : np.ndarray
            Scalar field values at the grid points.
        x : np.ndarray
            Coordinates of the grid points.
        '''
        ...
        

class SampledScalarField2D:

    def __init__(self, f: np.ndarray, x: np.ndarray, y: np.ndarray):
        '''
        Initialize a 2D scalar field.
        Parameters
        ----------
        f : np.ndarray
            Scalar field values at the grid points.
        x : np.ndarray
            X-coordinates of the grid points.
        y : np.ndarray
            Y-coordinates of the grid points.

        Notes
        -----
        The shape of f must be (len(x), len(y))
        and f[i, j] corresponds to the scalar value at (x[i], y[j]).

        '''
        ...


class SampledScalarField3D:

    def __init__(self, f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        '''
        Initialize a 3D scalar field.
        Parameters
        ----------
        f : np.ndarray
            Scalar field values at the grid points.
        x : np.ndarray
            X-coordinates of the grid points.
        y : np.ndarray
            Y-coordinates of the grid points.
        z : np.ndarray
            Z-coordinates of the grid points.

        Notes
        -----
        The shape of f must be (len(x), len(y), len(z))
        and f[i, j, k] corresponds to the scalar value at (x[i], y[j], z[k]).

        '''
        ...


class DelaunayTriangulation:

    @property
    def ndim(self)->int:
        '''
        Dimensionality of the triangulation.
        '''
        ...
        
    @property
    def npoints(self)->int:
        '''
        Number of scattered points.
        '''
        ...

    @property
    def nsimplices(self)->int:
        '''
        Number of simplices in the triangulation.
        '''
        ...


    @property
    def points(self)->np.ndarray[float]:
        '''
        Coordinates of the scattered points. Shape: (npoints, ndim).
        '''
        ...

    @property
    def simplices(self)->np.ndarray[int]:
        '''
        Indices of the points forming each simplex. Shape: (nsimplices, ndim+1).
        Each row contains the indices of the points that form a simplex.
        '''
        ...
    

    def find_simplex(self, coords: np.ndarray)->int:
        '''
        Get the index of the simplex containing the given coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the point to locate. Shape should be (n_dimensions,).

        Returns
        -------
        int
            Index of the simplex containing the point. Returns -1 if the point is outside the convex hull.
        '''
        ...

    def get_simplex(self, coords: np.ndarray)->np.ndarray[float]:
        '''
        Get the vertices of the simplex containing the given coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the point to locate. Shape should be (n_dimensions,).

        Returns
        -------
        np.ndarray
            Array of shape (ndim+1, ndim) containing the coordinates of the vertices for the simplex containing the point. Returns an empty array if the point is outside the convex hull.
        '''
        ...


class DelaunayTriND(DelaunayTriangulation):

    def __init__(self, points: np.ndarray):
        '''
        Initialize a Delaunay triangulation for scattered points in n-dimensional space.

        Parameters
        ----------
        points : np.ndarray
            Coordinates of the scattered points. Shape should be (n_points, n_dimensions).
        '''
        ...


class DelaunayTri1D(DelaunayTriangulation):

    def __init__(self, points: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...


class DelaunayTri2D(DelaunayTriangulation):

    def __init__(self, points: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...


class DelaunayTri3D(DelaunayTriangulation):

    def __init__(self, points: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...


class ScatteredScalarFieldND:

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
    def __init__(self, tri: DelaunayTriND, values: np.ndarray):
        '''
        Initialize a scattered scalar field from a Delaunay triangulation and corresponding scalar values.

        Parameters
        ----------
        tri : DelaunayTriND
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
    def tri(self)->DelaunayTriND:
        '''
        Delaunay triangulation of the scattered points.
        '''
        ...
        

    def __call__(self, *args: float)->float:
        '''
        Get the interpolated scalar field value at a specific point in n-dimensional space.

        Parameters
        ----------
        *args : float
            Coordinates of the point at which to evaluate the scalar field. The number of arguments should match the dimensionality of the points.

        Returns
        -------
        float
            Interpolated scalar field value at the given coordinates.
        '''
        ...


class ScatteredScalarField1D:

    @overload
    def __init__(self, points: np.ndarray, values: np.ndarray):
        ...

    @overload
    def __init__(self, tri: DelaunayTri1D, values: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...

    @property
    def values(self)->np.ndarray:
        ...

    @property
    def tri(self)->DelaunayTri1D:
        ...

    def __call__(self, x: float)->float:
        ...
        

class ScatteredScalarField2D:

    @overload
    def __init__(self, points: np.ndarray, values: np.ndarray):
        ...

    @overload
    def __init__(self, tri: DelaunayTri2D, values: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...

    @property
    def values(self)->np.ndarray:
        ...

    @property
    def tri(self)->DelaunayTri2D:
        ...

    def __call__(self, x: float, y: float)->float:
        ...


class ScatteredScalarField3D:

    @overload
    def __init__(self, points: np.ndarray, values: np.ndarray):
        ...

    @overload
    def __init__(self, tri: DelaunayTri3D, values: np.ndarray):
        ...

    @property
    def points(self)->np.ndarray:
        ...

    @property
    def values(self)->np.ndarray:
        ...

    @property
    def tri(self)->DelaunayTri3D:
        ...

    def __call__(self, x: float, y: float, z: float)->float:
        ...
