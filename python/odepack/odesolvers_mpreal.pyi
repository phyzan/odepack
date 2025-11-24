"""
Type stubs for arbitrary precision ODE solvers (odesolvers_mpreal backend).

This module provides template instantiations for MPFR arbitrary precision.
All classes are specialized versions of the generic templates in odesolvers_base.
"""

from __future__ import annotations
from .odesolvers_base import AbstractEvent, AbstractOdeResult, AbstractOdeSolution, AbstractOdeSolver, AbstractLowLevelODE, AbstractVariationalLowLevelODE, AbstractRK23, AbstractRK45, AbstractDOP853, AbstractBDF, AbstractPreciseEvent, AbstractPeriodicEvent, AbstractLowLevelFunction
import mpmath

# Type instantiation names for double precision backend
Event_MpReal = AbstractEvent[mpmath.mpf]
PreciseEvent_MpReal = AbstractPreciseEvent[mpmath.mpf]
PeriodicEvent_MpReal = AbstractPeriodicEvent[mpmath.mpf]
OdeResult_MpReal = AbstractOdeResult[mpmath.mpf]
OdeSolution_MpReal = AbstractOdeSolution[mpmath.mpf]
OdeSolver_MpReal = AbstractOdeSolver[mpmath.mpf]
RK23_MpReal = AbstractRK23[mpmath.mpf]
RK45_MpReal = AbstractRK45[mpmath.mpf]
DOP853_MpReal = AbstractDOP853[mpmath.mpf]
BDF_MpReal = AbstractBDF[mpmath.mpf]
LowLevelODE_MpReal = AbstractLowLevelODE[mpmath.mpf]
VariationalLowLevelODE_MpReal = AbstractVariationalLowLevelODE[mpmath.mpf]
LowLevelFunction_MpReal = AbstractLowLevelFunction[mpmath.mpf]
