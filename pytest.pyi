from __future__ import annotations

import numpy as np

class OdeResult:

    t: np.ndarray
    y: np.ndarray
    diverges: bool
    is_stiff: bool
    runtime: float


class LowLevelODE:

    def __init__(self, f):...

    def solve(self, ics: tuple, t: float, dt:float, **kwargs)->OdeResult:...

    def copy(self)->LowLevelODE:...
