


#include "ode/SolverDispatcher.hpp"
#include "ode/OdeInt.hpp"
#include "ode/Chaos/VariationalSolvers.hpp"
#include "ode/Interpolation/SampledVectorfields.hpp"

/**
 * @file odepack.h
 * @brief Main include file for odepack library.
 *
 * This file includes all necessary headers to use the odepack library for
 * solving ordinary differential equations (ODEs). It provides access to
 * various solver implementations, tools, and utilities.
 *
 * @author Foivos Zanias
 *
 * @note Make sure to include this file in your project to utilize odepack's
 *       functionalities.
 * @note Compile with -DNO_ODE_WARN to turn off unnecessary warnings that the solvers may throw on the console
 */