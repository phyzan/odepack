import numpy as np
from typing import overload

class NdInterpolator:

    @property
    def ndim(self)->int:...

    def __call__(self, *args: float)->np.ndarray:...


class RegularGridInterpolator(NdInterpolator):

    def __init__(self, values: np.ndarray, *args: np.ndarray):...


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
    def points(self)->np.ndarray[float]:...

    @property
    def values(self)->np.ndarray[float]:...

    @property
    def tri(self)->DelaunayTri:...

