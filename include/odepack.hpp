#ifndef ODEPACK_HPP
#define ODEPACK_HPP


#include "ode/Tools_impl.hpp"
#include "ode/SolverDispatcher_impl.hpp"
#include "ode/OdeInt_impl.hpp"
#include "ode/SolverState_impl.hpp"

#include "ode/Core/SolverBase_impl.hpp"
#include "ode/Core/RichBase_impl.hpp"
#include "ode/Core/VirtualBase_impl.hpp"
#include "ode/Core/Events_impl.hpp"

#include "ode/OdeResult/OdeResult_impl.hpp"

#include "ode/Interpolation/GridInterp_impl.hpp"
#include "ode/Interpolation/LinearNdInterpolator_impl.hpp"
#include "ode/Interpolation/SampledVectorfields_impl.hpp"
#include "ode/Interpolation/StateInterp_impl.hpp"

#include "ode/Solvers/Solvers_impl.hpp"

#include "ode/Chaos/VariationalSolvers_impl.hpp"



/**
 * @file odepack.hpp
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

#endif // ODEPACK_HPP