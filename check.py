import numpy as np
from pytest import *
from scipy.integrate import solve_ivp
import time


def f(t, y, args):
    return [y[2], y[3], -y[0], -y[1]]

ode = LowLevelODE(f)

ics = (0, [1, 1, 2.3, 4.5])

res1 = ode.solve(ics, 1000, 0.01, rtol=1e-5, atol=1e-10)

t1 = time.time()
res2 = solve_ivp(f, (0, 1000), ics[1], rtol=1e-5, atol=1e-10, args=(1,), first_step=0.01)
t2 = time.time()

print((t2-t1)/res1.runtime)

# print()