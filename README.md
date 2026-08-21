<p align="center">
<img src="https://img.shields.io/badge/C%2B%2B20-blue?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++20">
<img src="https://img.shields.io/badge/Header_Only-green?style=for-the-badge" alt="Header Only">
</p>

<h1 align="center">OdePack</h1>

<p align="center">
  <strong>A Modern C++ Library for Ordinary Differential Equations</strong>
</p>

<p align="center">
  High-performance, templated ODE solvers with flexible event detection mechanisms
</p>

---

## Overview

OdePack is a modern, object-oriented C++ header library for solving **Ordinary Differential Equations (ODEs)**. Originally inspired by Alan Hindmarsh's classic Fortran77 library, this implementation brings a fresh, template-based design heavily influenced by SciPy's ODE solver interface.

## Features:

- **Header-only**: No compilation needed, just include and use
- **Event system**: Detect and respond to user-defined conditions during integration
- **Dense output**: Smooth interpolation between computed steps
- **Memory efficient**: Solvers preallocate memory for zero heap (de)allocations between steps
- **Flexible solver policies**: Choose between static, rich, virtual, and rich-virtual solvers for performance vs. flexibility trade-offs
- **Extensible**: Easily add new solvers or event types
- **Template-based**: Allows for any numeric type including arbitrary precision (MPFR), supports automatic differentiation via [xdiff](https://github.com/phyzan/xdiff), lazy evaluation via [lazy](https://github.com/phyzan/lazy), and more
- **Dynamical systems analysis**: Built-in support for variational equations and Lyapunov exponent calculations

---

## Solvers

The library focuses on integrating systems of ODEs through objects that once instantiated, preallocate memory for any process that might be encountered,
and can be advanced in time while only updating their state in-place, without storing any integration history (if not requested).
This is particularly useful for long-running simulations, where memory usage and performance are critical.

Essentially, a solver object *iterates* over the solution of a system with a predefined accuracy, providing explicit
control over the integration process, and allowing for event detection and interpolation between steps (dense output).

This is achieved by defining a common interface for all solvers through the `BaseSolver` class, which is then specialized for any integration algorithm:

```cpp
template<typename Derived, typename T, size_t N, ode::SolverPolicy SP, ode::hasRhsFunc<T> OdeType>
class BaseSolver;
```

where:
- `Derived` is the derived solver class (CRTP)
- `T` is the numeric type (e.g., `double`, `float`, `mpfr::mpreal`)
- `N` is the number of equations in the system (`N > 0` for size known at compile-time, `N=0` for dynamic size using heap allocation)
- `SP` is the solver policy (see below)
- `OdeType` is the type of the ODE function (must satisfy `hasRhsFunc<T>` concept)

### SolverPolicy (SP) template parameter

The `SolverPolicy` template parameter controls inheritance and feature availability:

| Policy | Virtual | Events | Use Case |
|--------|---------|--------|---------------|
| `Static` | No | No | Maximum performance, compile-time type |
| `RichStatic` | No | Yes | Events needed, type known at compile-time |
| `Virtual` | Yes | No | Runtime solver selection, no events |
| `RichVirtual` | Yes | Yes | Full flexibility at runtime |


### OdeType template parameter

The `OdeType` template parameter must satisfy the `hasRhsFunc<T>` concept, which requires the ODE type to expose an `Rhs` member computing dq/dt:

```cpp
void Rhs(T* out, const T& t, const T* q);
```

- `out` — output array, receives dq/dt
- `t` — independent variable (time)
- `q` — current state array

An analytic Jacobian can optionally be provided via the `N x N` Jacobian matrix function
```cpp
void Jac(T* jac_mat, const T& t, const T* q);
```
which satisfies the `hasJacFunc<T>` concept, and is filled in column-major order as
```cpp
jac_mat[i + j*system_size] = df_i/dx_j
```

**Automatic differentiation**: if no analytic `Jac` is given but `Rhs` is written generically enough (templated), ideally as
```cpp
void Rhs(auto* out, const auto& t, const auto* q);
```
to also run over `xdiff::Dual` numbers (checked via `supportsDualRhs`), the library seeds the state with dual numbers and differentiates `Rhs` itself to obtain an exact Jacobian at no extra coding cost — no finite-difference approximation needed. Jacobian source is chosen automatically: exact analytic `Jac` > autodiff via `xdiff` > finite-difference approximation.

---

The `BaseSolver` class provides a common interface for all solvers, with the following main methods:

```cpp
// Accessors
void                Rhs(T* dq_dt, const T& t, const T* q) const; // Compute the right-hand side of the ODE system
void                Jac(T* J, const T& t, const T* q) const; // Compute the Jacobian of the ODE system (optional)
const T&            t() const; // Get the current time
View1D<T, N>        vector() const; // Get the current state vector
State<T>            ics() const; // Get the initial conditions
bool                is_running() const; // Check if the solver is still running
bool                is_dead() const; // Check if the solver cannot advance further
bool                diverges() const; // Check if the solver has diverged (nan/inf detected)
void                interp(T* out, const T& t) const; // Interpolate the solution at a given time within the old and new step interval
const std::string&  status() const; // Get the solver's status message

// Modifiers
bool                advance(); // Advance the solver by one step (automatic step size control)
bool                advance_until(const T& time); // Advance the solver until a specified time is reached
bool                advance_until(const T& time, const Callable& observer); // Advance the solver until a specified time is reached, calling an observer function at each step
bool                set_ics(T t0, const T* y0, T stepsize, int direction); // Set new initial conditions and reset the solver in-place (no memory reallocation happens)
void                Reset(); // Reset the solver to its initial state
BoxedInterp<T, N>   interpolate_until(const T& time, const Callable& observer = nullptr); // Advance the solver until a specified time is reached, returning an interpolator over the integration interval
```

### Available Solvers

Currently, the following solver classes are provided, overriding the proper `BaseSolver` methods for their respective algorithms:

| Solver | Type | Order | Description |
|--------|------|-------|-------------|
| `Euler` | Explicit | 1 | Basic Euler method |
| `RK23` | Explicit | 2/3 | Runge-Kutta 2(3) with adaptive stepping |
| `RK45` | Explicit | 4/5 | Dormand-Prince method (recommended for most problems) |
| `DOP853` | Explicit | 8 | High-order method with excellent dense output |
| `BDF` | Implicit | 1-5 | Backward Differentiation Formula for stiff problems |
| `RK4` | Explicit | 4 | Classic Runge-Kutta method with fixed step size |

## Event Detection

One main component of the library is the event detection system, which allows users to define conditions that trigger during integration. This feature was mainly developed for accurately detecting crossings in a *Poincaré surface of section* in dynamical systems, but it can be used for any situation where you need to detect when a certain condition is met during the integration of a system of ODE's.

- **Compile-time events**: For events that can be hardcoded in a project, it is preferred to use the compile-time event system, which avoids the overhead of virtual function calls, and allows for inlining and more compiler optimizations. This is achieved via the `ObjectiveSolver` class. See the relevant [example](tutorials/CompileTimeEvents.cpp) for different ways to declare relevant solvers.

- **Runtime events**: For events whose number or type is not known at compile-time, the polymorphic `Event<T>` class
is provided, which requires that the solver is declared with `ode::SolverPolicy::RichVirtual` or `RichStatic`.
See the relevant [example](tutorials/RuntimeEvents.cpp) for examples of how to use the runtime event system.

## Arbitrary Precision Support

All classes are templated, and the `T` template parameter can be any numeric type, including arbitrary precision:
```cpp
#include <odepack/odepack.hpp>
#include <mpreal.h>

using namespace ode;

int main(){

    // Set precision to 100 bits for all subsequent mpreal objects
    mpfr::mpreal::set_default_prec(100);

    using T = mpfr::mpreal; // `T` alias for simplicity
    std::array<T, 2> q0 = {10, 0}; // Initial conditions
    
    // Let's create a solver for the simple harmonic oscillator using the RK45 method
    pbox::Box<OdeSolver<T, 2>> solver = make_vsolver(
        Integrator::RK45,
        OdeData{
            .Rhs=[](auto* dq_dt, const auto& t, const auto* q){
                dq_dt[0] = q[1];
                dq_dt[1] = -q[0];
            },
        },
        T{0}, // t0
        View1D<T, 2>{q0.data()},
        T{1e-10}, // relative tolerance
        T{1e-10} // absolute tolerance
    );

    solver->do_advance_until(1000);
    const T& x = solver->get_vector()[0];
    const T& v = solver->get_vector()[1];
    std::cout << "Final state: " << x << ", " << v << std::endl;
    return 0;
}
```

However, `mpfr::mpreal` performs heap allocation when instantiated, and every intermediate algebraic expression
creates a temporary `mpreal` object. This can be avoided by using the `lazy` library, which allows for lazy evaluation of expressions and avoids unnecessary temporaries. See the [lazy](https://github.com/phyzan/lazy) submodule for more details. In practice, it can be used exactly like `mpreal` in most cases, by simply replacing `mpfr::mpreal` with `lazy::LazyType<mpfr::mpreal>` in the code above, using
```cpp
#include <lazy/apps/mpfrLazy.hpp>
```
and calling
```cpp
lazy::set_default_mpreal_prec(prec);
```
instead of
```cpp
mpfr::mpreal::set_default_prec(prec);
```

For instance, this [example](tutorials/CompileTimeEvents.cpp) demonstrates the performance difference between `mpreal` and `lazy::LazyType<mpfr::mpreal>` for a simple harmonic oscillator.

Note that as the number of requested bits of precision increases, the performance difference diminishes,
and the overhead of algebraic evaluations dominates.

# Installation


## Prerequisites

The prebuilt mpfr and gmp libraries are required for arbitrary precision support. These must be installed separately:

```bash
sudo apt install libmpfr-dev libgmp-dev
```

The rest of the external header-only dependencies are included as submodules in the `external/` directory, and they must be initialized and updated with the following command:

```bash
git submodule update --init --recursive
```

**Requirements:**
- C++20 compatible compiler

## C++ / CMake

### Macros

CMake options that toggle preprocessor macros across the library, its bundled dependencies, and the `odepack_tests` executable:

| CMake Option | Macro | Effect |
|--------------|-------|--------|
| `DPK_DENSE_RK4` | `RK4_DENSE` | Enable accurate RK4 dense output for the `RK4` solver, at the cost of additional memory usage and slightly slower performance. |
| `DPK_NO_WARN` | `NO_ODE_WARN` | Disable ODE solver console warnings. |
| `DPK_NO_NAN_CHECK` | `DPK_NO_NAN_CHECK` | Disable NaN/inf checks on solver output, for performance. |
| `DEBUG` | — | Debug build: `-O0 -g3 -ggdb3 -fno-omit-frame-pointer -UNDEBUG` (asserts enabled), instead of the default optimized release build (`-O3 -DNDEBUG`, LTO where supported). Also triggered by `-DCMAKE_BUILD_TYPE=Debug`. |
| `ODEPACK_BUILD_TESTS` | — | Build the `odepack_tests` executable from `tests/src/*.cpp`. Defaults to `ON` when configuring odepack directly, `OFF` when pulled in via `add_subdirectory` by another project. |

See useful [macros](https://github.com/phyzan/xdiff#macros) for the `xdiff` submodule.

## Linking via CMake

```cmake
add_subdirectory(path/to/odepack)
target_link_libraries(your_target PRIVATE odepack::odepack)
```
This gives you `<odepack/...>`, `<xdiff/...>` etc. includes, the required C++20 standard, and [macros](#macros) (toggle with e.g. `-D<MACRO_NAME>=ON`)

**Building and running the test suite:**
```bash
cmake -S . -B build
cmake --build build
./build/odepack_tests
```
or, with some macros enabled at configure time:
```bash
cmake -S . -B build -DDEBUG=ON
cmake --build build
```

---


## Architecture

The library uses a **two-tier architecture** combining static and dynamic polymorphism via CRTP (Curiously Recurring Template Pattern):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIRTUAL INTERFACE LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌──────────────────┐         ┌─────────────────────┐                 │
│        │  OdeSolver<T,N>  │────────▶│ OdeRichSolver<T,N>  │                 │
│        └──────────────────┘         └─────────────────────┘                 │
│         (base interface)             (+ runtime events)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATIC IMPLEMENTATION LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌────────────────────┐       ┌───────────────────────┐               │
│        │ BaseSolver<T,N,SP> │──────▶│  RichSolver<T,N,SP>   │               │
│        └────────────────────┘       └───────────────────────┘               │
│              (CRTP base) │             (+ runtime events)                   │
│                   │                             │                           │
│                   │                             │                           │
│                   ───────────────────────────────                           │
│                                  │                                          │
│        ┌─────────────────────────┼────────────────────────┐                 │
│        │            │            │            │           │                 │
│     ┌───▼───┐  ┌────▼───┐   ┌────▼───┐   ┌────▼───┐   ┌───▼───┐             │
│     │ Euler │  │  RK23  │   │  RK45  │   │ DOP853 │   │  BDF  │             │
│     └───────┘  └────────┘   └────────┘   └────────┘   └───────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
│             │   (events, no virtuals)                                       │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│  Virtual    │   RK45 ───▶ BaseSolver ───▶ OdeSolver                         │
│             │   (runtime polymorphism, no events)                           │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│ RichVirtual │   RK45 ───▶ RichSolver ───▶ BaseSolver ───▶ OdeRichSolver     │
│             │   (full features: virtuals + events)                          │
│             │                                                               │
└─────────────┴───────────────────────────────────────────────────────────────┘
```


### Design Patterns

| Pattern | Usage |
|---------|-------|
| **CRTP** | `BaseSolver<Derived, ...>` enables static dispatch without virtual overhead |
| **Policy Pattern** | `SolverPolicy` enum for compile-time feature selection |
| **Factory Pattern** | `getSolver()` and `make_solver()` for solver instantiation |

---

## Directory Structure

```
odepack/
├── include/
│   └── odepack/                     # All headers under odepack namespace, header-only
│       ├── ode/                     # Core ODE library
│       │   ├── Core/                # Foundation & base classes
│       │   │   ├── VirtualBase.hpp  # Virtual interfaces & solver policies
│       │   │   ├── VirtualTraits.hpp # Traits for virtual solvers
│       │   │   ├── SolverBase.hpp   # CRTP base solver
│       │   │   ├── RichBase.hpp     # Event-aware solver extension
│       │   │   ├── Events.hpp       # Event detection system
│       │   │   ├── FinDiff.hpp      # Finite difference utilities
│       │   │   ├── ObjectiveSolver.hpp  # Objective-based solver interface
│       │   │   └── *_impl.hpp       # Implementation files
│       │   │
│       │   ├── Solvers/             # Concrete solver implementations
│       │   │   ├── Solvers.hpp      # Common solver includes
│       │   │   ├── Euler.hpp        # Simple Euler method (1st order)
│       │   │   ├── RungeKutta.hpp   # Generic Runge-Kutta framework
│       │   │   ├── DOPRI.hpp        # Runge-Kutta RK23, RK45 (adaptive)
│       │   │   ├── DOP853.hpp       # High-order explicit RK (8th order)
│       │   │   ├── BDF.hpp          # Implicit solver for stiff systems
│       │   │   └── *_impl.hpp       # Implementation files
│       │   │
│       │   ├── Interpolation/       # Dense output & interpolation
│       │   │   ├── NdInterpolator.hpp   # N-dimensional interpolator base
│       │   │   ├── VectorFields.hpp # Sampled vector field interpolation
│       │   │   ├── Regular/         # Regular grid interpolation
│       │   │   │   ├── Grids.hpp    # Grid data structures
│       │   │   │   └── RegularGridInterpolator.hpp
│       │   │   ├── Scattered/       # Scattered data interpolation
│       │   │   │   ├── Delaunay.hpp # Delaunay triangulation
│       │   │   │   └── ScatteredNdInterpolator.hpp
│       │   │   ├── Univariate/      # 1D interpolation
│       │   │   │   └── StateInterp.hpp  # State interpolation for solvers
│       │   │   └── *_impl.hpp       # Implementation files
│       │   │
│       │   ├── Chaos/               # Dynamical systems analysis
│       │   │   ├── VariationalSolvers.hpp    # Lyapunov exponent computation
│       │   │   └── VariationalSolvers_impl.hpp
│       │   │
│       │   ├── OdeResult/           # Integration result storage
│       │   │   ├── OdeResult.hpp    # Result container
│       │   │   └── OdeResult_impl.hpp   # Implementation
│       │   │
│       │   ├── IntegratorEnum.hpp   # enum class of implemented integrators
│       │   ├── OdeInt.hpp           # High-level ODE wrapper
│       │   ├── SolverDispatcher.hpp # Factory for solver instantiation
│       │   ├── SolverState.hpp      # Solver state & status reporting
│       │   └── Tools.hpp            # Utilities, shared concepts & OdeData
│       │
│       ├── odepack.hpp              # Main C++ include (all headers)
│       └── odepackDecl.hpp          # Forward declarations
│
├── external/                        # Git submodules (bundled header-only dependencies)
│   ├── xdiff/                       # Automatic differentiation library (bundles its own `lazy` + `mpreal` submodules)
│   ├── ndspan/                      # Multi-dimensional array views and utilities
│   ├── polybox/                     # Wrapper for dynamically allocated types
│   └── qhull/                       # Convex hull library (for Delaunay triangulation)
│
├── tests/                           # C++ test suite, compiled into one odepack_tests executable
│   ├── include/                     # One <name>.hpp per test file, declaring void test_<name>()
│   └── src/                         # One <name>.cpp per test file, implementing it; main.cpp calls them all
│
├── tutorials/                       # Standalone example programs referenced from the README
│   ├── CompileTimeEvents.cpp        # Compile-time event system usage
│   └── RuntimeEvents.cpp            # Runtime (polymorphic) event system usage
│
├── .clang-tidy                      # clang-tidy check configuration
├── CMakeLists.txt                   # odepack::odepack interface target + odepack_tests build
├── LICENSE
└── README.md
```

---

## Performance Tips

- **Prefer SolverPolicy::Static**  when no runtime-event detection or type-erasure is required 
- **Set appropriate tolerances** - tighter tolerances mean smaller steps
- **Use `BDF`** for stiff problems


---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with modern C++ for scientists and engineers</sub>
</p>
