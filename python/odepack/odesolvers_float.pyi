"""
Type stubs for single precision ODE solvers (odesolvers_float backend).

This module provides template instantiations for single precision (np.float32).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import Event, OdeResult, OdeSolution, OdeSolver, LowLevelODE, VariationalLowLevelODE, RK23, RK45, DOP853, BDF

# Type instantiation names for float precision backend
Event_Float = Event[np.float32]
PresiceEvent_Float = Event[np.float32]
PeriodicEvent_Float = Event[np.float32]
OdeResult_Float = OdeResult[np.float32]
OdeSolution_Float = OdeSolution[np.float32]
OdeSolver_Float = OdeSolver[np.float32]
RK23_Float = RK23[np.float32]
RK45_Float = RK45[np.float32]
DOP853_Float = DOP853[np.float32]
BDF_Float = BDF[np.float32]
LowLevelODE_Float = LowLevelODE[np.float32]
VariationalLowLevelODE_Float = VariationalLowLevelODE[np.float32]
