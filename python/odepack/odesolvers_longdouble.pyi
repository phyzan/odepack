"""
Type stubs for extended precision ODE solvers (odesolvers_longdouble backend).

This module provides template instantiations for extended precision (np.longdouble).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import AbstractEvent, AbstractOdeResult, AbstractOdeSolution, AbstractOdeSolver, AbstractLowLevelODE, AbstractVariationalLowLevelODE, AbstractRK23, AbstractRK45, AbstractDOP853, AbstractBDF, AbstractPreciseEvent, AbstractPeriodicEvent, AbstractLowLevelFunction

# Type instantiation names for double precision backend
Event_LongDouble = AbstractEvent[np.longdouble]
PreciseEvent_LongDouble = AbstractPreciseEvent[np.longdouble]
PeriodicEvent_LongDouble = AbstractPeriodicEvent[np.longdouble]
OdeResult_LongDouble = AbstractOdeResult[np.longdouble]
OdeSolution_LongDouble = AbstractOdeSolution[np.longdouble]
OdeSolver_LongDouble = AbstractOdeSolver[np.longdouble]
RK23_LongDouble = AbstractRK23[np.longdouble]
RK45_LongDouble = AbstractRK45[np.longdouble]
DOP853_LongDouble = AbstractDOP853[np.longdouble]
BDF_LongDouble = AbstractBDF[np.longdouble]
LowLevelODE_LongDouble = AbstractLowLevelODE[np.longdouble]
VariationalLowLevelODE_LongDouble = AbstractVariationalLowLevelODE[np.longdouble]
LowLevelFunction_LongDouble = AbstractLowLevelFunction[np.longdouble]