"""
Type stubs for arbitrary precision ODE solvers (odesolvers_mpreal backend).

This module provides template instantiations for MPFR arbitrary precision.
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
import mpmath
from .odesolvers_base import Event, OdeResult, OdeSolution, OdeSolver, LowLevelODE, VariationalLowLevelODE, RK23, RK45, DOP853, BDF

# Type instantiation names for MPFR arbitrary precision backend
Event_MpReal = Event[mpmath.mpf]
PresiceEvent_MpReal = Event[mpmath.mpf]
PeriodicEvent_MpReal = Event[mpmath.mpf]
OdeResult_MpReal = OdeResult[mpmath.mpf]
OdeSolution_MpReal = OdeSolution[mpmath.mpf]
OdeSolver_MpReal = OdeSolver[mpmath.mpf]
RK23_MpReal = RK23[mpmath.mpf]
RK45_MpReal = RK45[mpmath.mpf]
DOP853_MpReal = DOP853[mpmath.mpf]
BDF_MpReal = BDF[mpmath.mpf]
LowLevelODE_MpReal = LowLevelODE[mpmath.mpf]
VariationalLowLevelODE_MpReal = VariationalLowLevelODE[mpmath.mpf]
