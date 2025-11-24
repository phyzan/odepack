"""
Type stubs for double precision ODE solvers (odesolvers_double backend).

This module provides template instantiations for double precision (np.float64).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import Event, OdeResult, OdeSolution, OdeSolver, LowLevelODE, VariationalLowLevelODE, RK23, RK45, DOP853, BDF

# Type instantiation names for double precision backend
Event_Double = Event[np.float64]
PreciseEvent_Double = Event[np.float64]
PeriodicEvent_Double = Event[np.float64]
OdeResult_Double = OdeResult[np.float64]
OdeSolution_Double = OdeSolution[np.float64]
OdeSolver_Double = OdeSolver[np.float64]
RK23_Double = RK23[np.float64]
RK45_Double = RK45[np.float64]
DOP853_Double = DOP853[np.float64]
BDF_Double = BDF[np.float64]
LowLevelODE_Double = LowLevelODE[np.float64]
VariationalLowLevelODE_Double = VariationalLowLevelODE[np.float64]
