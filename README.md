The ODE parameters and code philosophy closely resemble that of scipy's ODE and OdeSolver class on purpose, as some parts of the code (like the Runge-Kutta classes) have simply been translated from Python to C++.
However this implementation targets performance, parallelization, higher flexibility, progress displaying, and better support for Event encounters during an ode integration.

