"""
Type stubs for extended precision ODE solvers (odesolvers_longdouble backend).

This module provides template instantiations for extended precision (np.longdouble).
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import numpy as np
from .odesolvers_base import Event, OdeResult, OdeSolution, OdeSolver, LowLevelODE, VariationalLowLevelODE, RK23, RK45, DOP853, BDF

# Type instantiation names for long double precision backend
Event_LongDouble = Event[np.longdouble]
PresiceEvent_LongDouble = Event[np.longdouble]
PeriodicEvent_LongDouble = Event[np.longdouble]
OdeResult_LongDouble = OdeResult[np.longdouble]
OdeSolution_LongDouble = OdeSolution[np.longdouble]
OdeSolver_LongDouble = OdeSolver[np.longdouble]
RK23_LongDouble = RK23[np.longdouble]
RK45_LongDouble = RK45[np.longdouble]
DOP853_LongDouble = DOP853[np.longdouble]
BDF_LongDouble = BDF[np.longdouble]
LowLevelODE_LongDouble = LowLevelODE[np.longdouble]
VariationalLowLevelODE_LongDouble = VariationalLowLevelODE[np.longdouble]
