ODEPACK is a collection of solvers for ordinary differential equations, originally created for Fortran77 by Alan HindMarsh.
This is a modern object-oriented C++ header library that provides an easy-to-use interface for ODE integration, with support for extending the collection of core algorithms (e.g., RK45, BDF).
The design is heavily influenced by SciPy's implementation, resulting in a familiar class hierarchy for Python users.
The code relies on CRTP inheritance with a virtual frontend for the OdeSolver class, which acts as an iterator of the ODE's solution.
A key design goal is to support event handling during integration (detecting user-defined events) while maintaining high performance. As a header-only library, it allows solvers to be instantiated with any numerical type, such as mpreal for arbitrary precision arithmetic.

In practice, when defining an OdeSolver through a derived class like

```
RK45<double, 4> solver({ode}, t0, y0, rtol, atol, min_step, max_step, first_step, direction, extra_args, events);
```

the RK45 solver simply advances through the solution of the ODE by calling

```
solver.advance();
```

which uses the 5th-order accurate Runge-Kutta Dormand-Prince adaptive step method. Importantly, `advance()` only updates the solver's internal state and does not automatically store previous steps, minimizing memory usage.
The current time, state vector, and step size can be accessed via

```
solver.t();
solver.q();
solver.stepsize();
```

To automatically store the solution during integration, use the ODE class. An instance of this class manages the complete ODE solution and is updated via:

```
ode_result = ode.integrate(interval, time_frames_to_store, event_options);
```

The Python extension can be installed using
```
pip install ./python
```
in the root folder, and uses the same object oriented structure.
