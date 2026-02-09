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

# Installation


### Prerequisites

The library contains modules for interpolating sampled fields using the Q-Hull headers.
To exploit this functionality, install Q-Hull by running the following command.
For the Python installation, this is a required dependency.

```bash
sudo apt install libqhull-dev
```

To use the arbitrary precision features, install the C++ wrapper of MPFR and GMP:

```bash
sudo apt install libmpfrc++-dev
```


### C++ (Header-Only)

```bash
git clone https://github.com/phyzan/odepack.git
cd odepack
sudo make install
```

This installs headers to `/usr/local/include`. To use a custom location:
```bash
sudo make install PREFIX=/path/to/install
```

To uninstall:
```bash
sudo make uninstall
```

Then include in your code:
```cpp
#include <odepack/ODEPACK.hpp> 
```

**Requirements:**
- C++20 compatible compiler

### Python

```bash
pip install ./python
```

To install with arbitrary precision support, assuming that the C++ wrapper of MPFR and GMP is installed (see above), run
```bash
CMAKE_ARGS="-DMPREAL=ON" pip install ./python
```

**Requirements:**
- Python 3.12+

There are optional build flags for the Python bindings:

- MPREAL: Enable arbitrary precision support
- DEBUG: Enable debug build
- RK4_DENSE: Enable accurate RK4 dense output for the RK4 solver, with the cost of additional memory usage and slightly slower performance.

Use any of them by setting the `CMAKE_ARGS` environment variable before installation, and adding the `-D` character before the flag, for example:

For a standard debug build, run
```bash
CMAKE_ARGS="-DDEBUG=ON" pip install ./python
```

or for a debug build with arbitrary precision support, run
```bash
CMAKE_ARGS="-DDEBUG=ON -DMPREAL=ON" pip install ./python
```

---

## Quick Start

### C++ Example

```cpp
#include <odepack/ODEPACK.hpp>

using namespace ode;

int main() {

    // Initial conditions
    double t = 0.0;
    std::array<double, 2> y0 = {3.0, 0.0};

    // Define the y' = 1 crossing
    PreciseEvent<double> event(
        "event",
        [](const double& t, const double* y, const double* args, const void* ptr){
            return y[1] - 1.0;
        });


    // General signature for ODE function
    auto df_dt = [&](double* dy_dt, const double& t, const double* y, const double* args, const void* ptr) {
        //2D oscillator: y'' + y = 0  => y1' = y2, y2' = -y1
        dy_dt[0] = y[1];
        dy_dt[1] = -y[0];
    };

    // Solver policy determines the capabilities of the solver,
    // by slightly sacrificing performance. Here we use RichStatic
    // which allows event detection.
    constexpr SolverPolicy SP = SolverPolicy::RichStatic;

    // Create solver
    auto solver = getSolver<RK45, double, 2, SP>(
        OdeData{.rhs=df_dt},   // ODE function
        t,                 // Initial time
        y0.data(),
        2,             // ODE system size
        1e-6,          // Relative tolerance
        1e-9,          // Absolute tolerance
        0.0,       // Minimum step size
        1.0,       // Maximum step size
        0.0,    // First step size (0 = auto)
        1,              // Integration direction
        {},            // Additional args to be passed to ODE function
        {&event} // Events

    );

    // Advance until event is detected
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

ode = odesys.get(t0=0, q0=[3.0, 0.0], rtol=1e-6, atol=1e-9, stepsize=0.01, compiled=True, scalar_type="double") #use False for pure python version

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
| `set_ics(t0, y0, stepsize, direction)` | Set new initial conditions |
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
#include <odepack/ODEPACK.hpp>

using namespace ode;

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
        OdeData{.rhs=df_dt},   // ODE function
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
#include <odepack/Core/Events.hpp>

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

ODEPACK uses a **two-tier architecture** combining static and dynamic polymorphism via CRTP (Curiously Recurring Template Pattern):

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
│              (CRTP base) │             (+ events/interpolation)             │
│                   │                             │                           │
│                   │                             │                           │
│                   ───────────────────────────────                           │
│                                  │                                          │
│         ┌────────────────────────┼──────────────────────────┐               │
│         │            │           │          │               │               │
│      ┌──▼───┐   ┌────▼───┐  ┌────▼───┐ ┌────▼──┐  ┌────▼──┐                 │
│      │Euler │   │  RK23  │  │  RK45  │ │DOP853 │  │  BDF  │                 │
│      └──────┘   └────────┘  └────────┘ └───────┘  └───────┘                 │
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

### Design Patterns

| Pattern | Usage |
|---------|-------|
| **CRTP** | `BaseSolver<Derived, ...>` enables static dispatch without virtual overhead |
| **Policy Pattern** | `SolverPolicy` enum for compile-time feature selection |
| **Factory Pattern** | `getSolver()` and `get_virtual_solver()` for solver instantiation |
| **Polymorphic Wrapper** | `PolyWrapper<T>` for type-erased ownership of interpolators & events |

---

## Directory Structure

```
odepack/
├── include/
│   ├── odepack/
│   │   ├── Core/                    # Foundation & base classes
│   │   │   ├── VirtualBase.hpp      # Virtual interfaces & solver policies
│   │   │   ├── SolverBase.hpp       # CRTP base solver
│   │   │   ├── RichBase.hpp         # Event-aware solver extension
│   │   │   └── Events.hpp           # Event detection system
│   │   │
│   │   ├── Solvers/                 # Concrete solver implementations
│   │   │   ├── Euler.hpp            # Simple Euler method (1st order)
│   │   │   ├── DOPRI.hpp            # Runge-Kutta RK23, RK45 (adaptive)
│   │   │   ├── DOP853.hpp           # High-order explicit RK (8th order)
│   │   │   └── BDF.hpp              # Implicit solver for stiff systems
│   │   │
│   │   ├── Interpolation/           # Dense output & interpolation
│   │   │   ├── Interpolators.hpp    # Base interpolator interface
│   │   │   └── GridInterp.hpp       # Grid & vector field interpolation
│   │   │
│   │   ├── Chaos/                   # Dynamical systems analysis
│   │   │   └── VariationalSolvers.hpp  # Lyapunov exponent computation
│   │   │
│   │   ├── odepack.hpp              # Main include (all headers)
│   │   ├── OdeInt.hpp               # High-level ODE wrapper
│   │   ├── SolverDispatcher.hpp     # Factory for solver instantiation
│   │   ├── SolverState.hpp          # Solver state & status reporting
│   │   └── Tools.hpp                # Utilities (PolyWrapper, etc.)
│   │
│   └── ndspan/                      # Multi-dimensional array library
│       ├── ndspan.hpp
│       ├── arrays.hpp
│       └── ...
│
├── python/
│   ├── src/                         # Python bindings (pybind11)
│   ├── odepack/                     # Python package
│   └── tests/                       # Python tests
│
├── tests/                           # C++ tests
├── LICENSE
└── README.md
```

### Component Overview

| Component | Description |
|-----------|-------------|
| **Core/** | Base classes, virtual interfaces, event system, and solver policies |
| **Solvers/** | Concrete integrator implementations (Euler, RK23, RK45, DOP853, BDF) |
| **Interpolation/** | Dense output providers and grid interpolation utilities |
| **Chaos/** | Specialized tools for variational equations and Lyapunov exponents |
| **OdeInt.hpp** | High-level `ODE<T,N>` wrapper for trajectory storage and result access |
| **SolverDispatcher.hpp** | Factory functions for solver instantiation |
| **Tools.hpp** | Utilities including `PolyWrapper` for polymorphic type ownership |

---

## Arbitrary Precision

ODEPACK supports arbitrary precision arithmetic via MPFR:

```cpp
#include <odepack/ODEPACK.hpp>

using namespace ode;

using mpfr::mpreal;

template<typename T>
void df_dt(T* dy_dt, const T& t, const T* y, const T* args, const void* ptr) {
    //2D oscillator: y'' + y = 0
    dy_dt[0] = y[1];
    dy_dt[1] = -y[0];
}

template<typename T>
T crossing(const T& t, const T* y, const T* args, const void* ptr) {
    return y[1] - 1;
}

int main() {

    // Initial conditions
    std::array<mpreal, 2> y0 = {3, 0};

    // Define the y' = 1 crossing
    PreciseEvent<mpreal> event("event", crossing<mpreal>);

    constexpr SolverPolicy SP = SolverPolicy::RichStatic;

    // Create solver
    mpreal t = 0;
    mpreal rtol = "1e-6";
    mpreal atol = "1e-9";
    mpreal min_step = 0;
    mpreal max_step = 1;
    mpreal stepsize = 0;
    constexpr size_t nsys = 2;
    int dir = 1;
    auto solver = getSolver<RK45, mpreal, nsys, SP>(
        OdeData{.rhs=df_dt<mpreal>},
        t,
        y0.data(),
        nsys,
        rtol,
        atol,
        min_step,
        max_step,
        stepsize,
        dir,
        {},
        {&event}
    );

    // Advance until event is detected
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

Compile with `-DMPREAL` flag and link against MPFR/GMP with `-lmpfr -lgmp`.

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
