import numpy as np
from typing import overload

class NdInterpolator:

    @property
    def ndim(self)->int:...

    def __call__(self, *args: np.ndarray)->np.ndarray:
        '''
        Interpolate the values at the given coordinates.

        Parameters
        ----------
        *args : np.ndarray
            Coordinates of the points to interpolate. The number of arguments must be equal to self.ndim. The shape of the input coordinates can be arbitrary, but all arguments must have the same shape

        Returns
        -------
        np.ndarray
            Interpolated values at the given coordinates. The shape of the output depends on the shape of the input coordinates and the shape of the values at the grid points.
            This is shape of input coordinates + shape of values at each grid points.

        Example
        -------
        If the values at each grid point have shape (2, 4, 5), and the input coordinates have shape (3, 3) on each argument, the output will have shape (3, 3, 2, 4, 5).
        '''
        ...
    

class RegularGridInterpolator(NdInterpolator):

    def __init__(self, values: np.ndarray, *args: np.ndarray):...

    @property
    def values(self)->np.ndarray[float]:
        '''
        The values at the grid points, with shape
        (n1, n2, ... n_ndim, *shape_per_grid_point)

        If the class was instanciated directly, the array is the
        same as the one passed in the constructor.
        '''
        ...

    @property
    def grid(self)->tuple[np.ndarray[float], ...]:
        '''
        The grid points for each dimension, as a tuple of 1D arrays.
        Each array contains the coordinates of the grid points along that dimension.
        '''
        ...


class DelaunayTri:

    def __init__(self, points: np.ndarray):...


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

    @property
    def total_volume(self)->float:
        '''
        Total volume of the triangulation in the parameter space.
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



class ScatteredNdInterpolator(NdInterpolator):

    @overload
    def __init__(self, points: np.ndarray, values: np.ndarray):...

    @overload
    def __init__(self, tri: DelaunayTri, values: np.ndarray):...

    @property
    def points(self)->np.ndarray[float]:
        '''
        Coordinates of the scattered points. Shape: (npoints, ndim).
        '''
        ...

    @property
    def values(self)->np.ndarray[float]:
        '''
        Values at the scattered points. Shape: (npoints, *shape_per_point).
        '''
        ...

    @property
    def tri(self)->DelaunayTri:
        '''
        Internal Delaunay triangulation object of the scattered points.
        This returns a reference and no copy is made, as such objects can be large.
        '''
        ...

