"""
Type stubs for double precision ODE solvers (odesolvers_double backend).

This module provides template instantiations for double precision (np.float64).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import AbstractEvent, AbstractOdeResult, AbstractOdeSolution, AbstractOdeSolver, AbstractLowLevelODE, AbstractVariationalLowLevelODE, AbstractRK23, AbstractRK45, AbstractDOP853, AbstractBDF, AbstractPreciseEvent, AbstractPeriodicEvent, AbstractLowLevelFunction

# Type instantiation names for double precision backend
Event_Double = AbstractEvent[np.float64]
PreciseEvent_Double = AbstractPreciseEvent[np.float64]
PeriodicEvent_Double = AbstractPeriodicEvent[np.float64]
OdeResult_Double = AbstractOdeResult[np.float64]
OdeSolution_Double = AbstractOdeSolution[np.float64]
OdeSolver_Double = AbstractOdeSolver[np.float64]
RK23_Double = AbstractRK23[np.float64]
RK45_Double = AbstractRK45[np.float64]
DOP853_Double = AbstractDOP853[np.float64]
BDF_Double = AbstractBDF[np.float64]
LowLevelODE_Double = AbstractLowLevelODE[np.float64]
VariationalLowLevelODE_Double = AbstractVariationalLowLevelODE[np.float64]
LowLevelFunction_Double = AbstractLowLevelFunction[np.float64]
