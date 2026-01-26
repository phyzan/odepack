<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++20">
  <img src="https://img.shields.io/badge/Header_Only-yes-green?style=for-the-badge" alt="Header Only">
  <img src="https://img.shields.io/badge/Python-3.12%2B-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">ODEPACK</h1>

<p align="center">
  <strong>A Modern C++ Library for Ordinary Differential Equations</strong>
</p>

<p align="center">
  High-performance, header-only ODE solvers with event detection, dense output, and Python bindings
</p>

---

## Overview

ODEPACK is a modern, object-oriented C++ header library for solving **Ordinary Differential Equations (ODEs)**. Originally inspired by Alan Hindmarsh's classic Fortran77 ODEPACK library, this implementation brings a fresh, template-based design heavily influenced by SciPy's ODE solver interface.

### Why ODEPACK?

| Feature | Benefit |
|---------|---------|
| **Header-only** | No compilation needed, just include and use |
| **Template-based** | Use any numeric type including arbitrary precision (MPFR) |
| **Event system** | Detect and respond to user-defined conditions during integration |
| **Dense output** | Smooth interpolation between computed steps |
| **Python bindings** | Seamless integration with Python via pybind11 |
| **Memory efficient** | Only stores what you need |

---

## Features

### Solvers

| Solver | Type | Order | Description |
|--------|------|-------|-------------|
| `Euler` | Explicit | 1 | Basic Euler method |
| `RK23` | Explicit | 2/3 | Runge-Kutta 2(3) with adaptive stepping |
| `RK45` | Explicit | 4/5 | Dormand-Prince method (recommended for most problems) |
| `DOP853` | Explicit | 8 | High-order method with excellent dense output |
| `BDF` | Implicit | 1-5 | Backward Differentiation Formula for stiff problems |

### Core Capabilities

- **Adaptive Step Size Control** - Automatic error estimation and step adjustment
- **Forward & Backward Integration** - Integrate in either time direction
- **Configurable Tolerances** - Set `rtol`, `atol`, `min_step`, `max_step`
- **Variational Equations** - Built-in support for Lyapunov exponent calculations
- **Parallel Integration** - OpenMP support for solving multiple ODE systems
- **Custom Solvers** - Extend the library with your own integration methods

### Event Detection

- **Zero-crossing detection** with bisection refinement
- **Periodic events** at fixed time intervals
- **State modification masks** for discontinuous changes
- **Event collections** for managing multiple events

---

## Installation

### C++ (Header-Only)

Simply include the headers in your project:

```cpp
#include <odepack/solvers.hpp>
```

**Requirements:**
- C++20 compatible compiler
- OpenMP (optional, for parallel integration)
- MPFR/GMP (optional, for arbitrary precision)

### Python

```bash
pip install ./python
```

**Requirements:**
- Python 3.12+

---

## Quick Start

### C++ Example

```cpp
#include <odepack/solvers.hpp>

void df_dt(double* dy_dt, const double& t, const double* y, const double* args, const void* obj) {
    //2D oscillator: y'' + y = 0
    dy_dt[0] = y[1];
    dy_dt[1] = -y[0];
}

int main() {

    // Initial conditions
    double t0 = 0.0;
    std::array<double, 2> y0 = {3.0, 0.0};

    // y' = 1 crossing
    ode::PreciseEvent<double> event(
        "event",
        [](const double& t, const double* y, const double* args, const void*){
            return y[1]-1;
        });

    // Create solver
    ode::RK45<double, 2, ode::SolverPolicy::RichStatic> solver(
        {.rhs=df_dt},   // ODE function
        t0,                 // Initial time
        y0.data(),
        2,             // ODE system size
        1e-6,          // Relative tolerance
        1e-9,          // Absolute tolerance
        0.0,       // Minimum step size (0 = auto)
        1.0,       // Maximum step size
        0.01,    // First step size
        1,              // Integration direction
        {},            // Additional args
        {&event} // Events

    );

    // Integrate to t = 5
    while (!solver.at_event()) {
        solver.advance();
    }

    std::cout << "Event detected at t = " << solver.t() << "\n";
    std::cout << "State at event: ";
    auto v = solver.vector();
    for (size_t i = 0; i < 2; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected state: {..., 1}" << std::endl;

    return 0;
}
```

### Python Example

```python
from odepack import *

t, x, y = symbols('t, x, y')

event = SymbolicPreciseEvent("event", y-1)

dq_dt = [y, -x]
odesys = OdeSystem(dq_dt, t, [x, y], events=[event])

ode = odesys.get(t0=0, q0=[3.0, 0.0], rtol=1e-6, atol=1e-9, first_step=0.01, compiled=True, scalar_type="double") #use False for pure python version

solver = ode.solver()   # get a copy of the internal solver
print(ode.__class__)    #LowLevelODE
print(solver.__class__) #RK45

while not solver.at_event:
    solver.advance()

print(f"Event detected at t={solver.t:.6f}")
print("State at event:", solver.q)
print("Expected state:", "[..., 1]")
```

---

# API Reference

## ODE Solver

### Solver classes only update their internal state at each step: no reallocation

| Method | Description |
|--------|-------------|
| `advance()` | Perform one adaptive integration step |
| `advance_until(time)` | Integrate until the specified time |
| `advance_until(func, tol, dir)` | Integrate until an event condition is met |
| `reset()` | Return to initial conditions |
| `set_ics(t0, y0, stepsize)` | Set new initial conditions |
| `interp(t)` | Interpolate solution at arbitrary time within last step |
| `clone()` | Create a dynamic copy of the solver |

### Common Properties

| Property | Description |
|----------|-------------|
| `t()` | Current time |
| `vector()` | Current state vector |
| `stepsize()` | Current step size |
| `at_event()` | If the solver state is currently at a detected event |

## ODE Class

### ODE classes use an internal solver to store integration history

```cpp
#include <odepack/ode.hpp>

using ode::RK45, ode::ODE, ode::OdeResult, ode::StepSequence, ode::PreciseEvent, ode::EventOptions, ode::EventMap;

void df_dt(double* dy_dt, const double& t, const double* y, const double* args, const void* obj) {
    //2D oscillator: y'' + y = 0
    dy_dt[0] = y[1];
    dy_dt[1] = -y[0];
}

int main() {

    // Initial conditions
    double t0 = 0.0;
    std::array<double, 2> y0 = {3.0, 0.0};

    // y' = 1 crossing
    PreciseEvent<double> event(
        "event",
        [](const double& t, const double* y, const double* args, const void*){
            return y[1]-1;
        });

    // Create ode
    ODE<double, 2> ode(
        {.rhs=df_dt},   // ODE function
        t0,                 // Initial time
        y0.data(),
        2,             // ODE system size
        1e-13,          // Relative tolerance
        1e-13,          // Absolute tolerance
        0.0,       // Minimum step size (0 = auto)
        1.0,       // Maximum step size
        0.01,    // First step size
        1,              // Integration direction
        {},            // Additional args
        {&event},// Events
        "RK45"       // Solver method, dynamically selected
    );

    EventOptions options{.name = "event", .max_events = 10, .terminate = true};

    // Integrate until the maximum number of events is reached
    StepSequence<double> steps_to_save; //default constructor means save all steps

    // Set a long integration interval to ensure we hit the event multiple times
    // The result stores all specified time steps, along with events encountered
    OdeResult<double, 2> result = ode.integrate(1000000, steps_to_save, {options});
    std::cout << "Integration completed at t = " << result.t().back() << "\n";
    std::cout << "Integration completed in " << result.runtime() << " seconds.\n";
    std::cout << "Number of time points: " << result.t().size() << "\n";
    std::cout << "Number of events detected: " << result.event_map().at("event").size() << "\n";
    std::cout << "Integration success: " << (result.success() ? "true" : "false") << "\n";
    std::cout << "Divergence detected: " << (result.diverges() ? "true" : "false") << "\n";
    std::cout << "Termination message: " << result.message() << "\n";

    // Extract the indices of the event occurrences
    const std::vector<size_t>& event_data = result.event_map().at("event");

    for (size_t i : event_data) {
        std::cout << "Event detected at t = " << result.t()[i] << "\n";
        std::cout << "State vector at event: " << result.q(i, 0) << ", " << result.q(i, 1) << std::endl;
    }

    return 0;
}
```

---

## Event System

ODEPACK features an event detection system that handles integration events, with special processing for multiple or simultaneous event detections, with or without discontinuities, at each time step.

### Zero-Crossing Events

```cpp
#include <odepack/events.hpp>

// Detect when y[0] crosses zero
PreciseEvent<double> event(
    "event",
    [](const double& t, const double* y, const double* args, const void*) -> double {
        return y[1]-1;
    }, //crossing at y[1] = 1
    1, //only cross when the sign change is from negative to positive
    [](double* y_new, const double& t, const double* y, const double* args, const void* obj) -> void {
        y_new[0] = 1;
        y_new[1] = -2.5;
    } // change the ODE state vector when the event is encountered    
    );
```

---

## Architecture

ODEPACK uses a **two-tier architecture** combining static and dynamic polymorphism:

### Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIRTUAL INTERFACE LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌──────────────────┐         ┌─────────────────────┐                 │
│        │  OdeSolver<T,N>  │────────▶│ OdeRichSolver<T,N>  │                 │
│        └──────────────────┘         └─────────────────────┘                 │
│         (base interface)             (+ events/interpolation)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATIC IMPLEMENTATION LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌────────────────────┐       ┌───────────────────────┐               │
│        │ BaseSolver<T,N,SP> │──────▶│  RichSolver<T,N,SP>   │               │
│        └────────────────────┘       └───────────────────────┘               │
│              (CRTP base) │             (+ events/interpolation)             |
│                   │                             │                           |
│                   │                             │                           |
|                   ───────────────────────────────                           |
│                                  │                                          |
│         ┌────────────────────────┼──────────────────────────┐               |
│         │            │           │          │               │               |
│         ┌────▼───┐  ┌─────▼────┐ ┌────▼────┐ ┌───▼───┐ ┌───▼───┐            |
│         │ Euler  │  │   RK23   │ │  RK45   │ │DOP853 │ │  BDF  │            |
│         └────────┘  └──────────┘ └─────────┘ └───────┘ └───────┘            |
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solver Policies (SP)

The `SolverPolicy` template parameter controls inheritance and feature availability:

```
┌─────────────┬───────────────────────────────────────────────────────────────┐
│   POLICY    │                      INHERITANCE CHAIN                        │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│   Static    │   RK45 ───▶ BaseSolver                                        │
│             │   (maximum performance, no virtuals, no events)               │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│ RichStatic  │   RK45 ───▶ RichSolver ───▶ BaseSolver                        │
│             │   (events + interpolation, no virtuals)                       │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│  Virtual    │   RK45 ───▶ BaseSolver ───▶ OdeSolver                         │
│             │   (runtime polymorphism, no events)                           │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│ RichVirtual │   RK45 ───▶ RichSolver ───▶ BaseSolver ───▶ OdeRichSolver     │
│             │   (full features: virtuals + events + interpolation)          │
│             │                                                               │
└─────────────┴───────────────────────────────────────────────────────────────┘
```

### Policy Summary

| Policy | Virtual | Events | Interpolation | Use Case |
|--------|---------|--------|---------------|----------|
| `Static` | No | No | No | Maximum performance, compile-time type |
| `RichStatic` | No | Yes | Yes | Events needed, type known at compile-time |
| `Virtual` | Yes | No | No | Runtime solver selection, no events |
| `RichVirtual` | Yes | Yes | Yes | Full flexibility at runtime |

---

## Directory Structure

```
odepack/
├── include/
│   ├── odepack/
│   │   ├── solverbase.hpp      # CRTP base class
│   │   ├── virtualsolver.hpp   # Virtual interfaces
│   │   ├── rich_solver.hpp     # Event-aware solver
│   │   ├── rk_adaptive.hpp     # Runge-Kutta base
│   │   ├── dop853.hpp          # 8th order RK
│   │   ├── bdf.hpp             # Implicit BDF method
│   │   ├── euler.hpp           # Euler method
│   │   ├── events.hpp          # Event detection system
│   │   ├── ode.hpp             # High-level ODE wrapper
│   │   ├── solvers.hpp         # Solver factory
│   │   ├── interpolators.hpp   # Dense output
│   │   └── variational.hpp     # Lyapunov exponents
│   └── ndspan/                 # Multi-dimensional array library
│       ├── ndspan.hpp
│       ├── arrays.hpp
│       └── ...
├── python/
│   ├── src/                    # Python bindings (pybind11)
│   ├── odepack/                # Python package
│   └── tests/                  # Python tests
├── LICENSE
└── README.md
```

---

## Arbitrary Precision

ODEPACK supports arbitrary precision arithmetic via MPFR:

```cpp
#include <mpreal.h>
#include <odepack/solvers.hpp>

using mpfr::mpreal;

// Set precision to 256 bits
mpreal::set_default_prec(256);

RK45<mpreal, 2> solver(
    {ode}, t0, y0,
    mpreal("1e-50"),  // rtol
    mpreal("1e-60")   // atol
);
```

Compile with `-DMPREAL` flag and link against MPFR/GMP.

---

## Performance Tips

1. **Use static solvers** (`RK45<T, N, SolverPolicy::Static>`) when no event detection is required
2. **Set appropriate tolerances** - tighter tolerances mean smaller steps
3. **Use `BDF`** for stiff problems

---

## Comparison with Other Libraries

| Feature | ODEPACK | Boost.Odeint | GSL | SciPy |
|---------|---------|--------------|-----|-------|
| Header-only | Yes | Yes | No | N/A |
| C++ Standard | C++20 | C++11 | C99 | N/A |
| Event detection | Yes | Limited | No | Yes |
| Dense output | Yes | Limited | No | Yes |
| Arbitrary precision | Yes | Yes | No | No |
| Python bindings | Yes | No | No | Native |

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Alan Hindmarsh** - Original ODEPACK Fortran library
- **SciPy Team** - Design inspiration for the Python-like interface
- **pybind11** - Seamless C++/Python bindings

---

<p align="center">
  <sub>Built with modern C++ for scientists and engineers</sub>
</p>
