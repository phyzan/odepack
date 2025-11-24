"""
Type stubs for single precision ODE solvers (odesolvers_float backend).

This module provides template instantiations for single precision (np.float32).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import AbstractEvent, AbstractOdeResult, AbstractOdeSolution, AbstractOdeSolver, AbstractLowLevelODE, AbstractVariationalLowLevelODE, AbstractRK23, AbstractRK45, AbstractDOP853, AbstractBDF, AbstractPreciseEvent, AbstractPeriodicEvent, AbstractLowLevelFunction

# Type instantiation names for double precision backend
Event_Float = AbstractEvent[np.float32]
PreciseEvent_Float = AbstractPreciseEvent[np.float32]
PeriodicEvent_Float = AbstractPeriodicEvent[np.float32]
OdeResult_Float = AbstractOdeResult[np.float32]
OdeSolution_Float = AbstractOdeSolution[np.float32]
OdeSolver_Float = AbstractOdeSolver[np.float32]
RK23_Float = AbstractRK23[np.float32]
RK45_Float = AbstractRK45[np.float32]
DOP853_Float = AbstractDOP853[np.float32]
BDF_Float = AbstractBDF[np.float32]
LowLevelODE_Float = AbstractLowLevelODE[np.float32]
VariationalLowLevelODE_Float = AbstractVariationalLowLevelODE[np.float32]
LowLevelFunction_Float = AbstractLowLevelFunction[np.float32]

