#ifndef ODEPACK_VIRTUAL_TRAITS_HPP
#define ODEPACK_VIRTUAL_TRAITS_HPP

#include "../Tools.hpp" // IWYU pragma: keep

namespace ode{


struct EmptySolver{};

template<typename T, size_t N>
class OdeSolver;

template<typename T, size_t N>
class OdeRichSolver;

enum class UtilPolicy : std::uint8_t{ Virtual, RichVirtual};

enum class SolverPolicy : std::uint8_t{ Static, RichStatic, Virtual, RichVirtual};


namespace detail{

template<typename T, size_t N, UtilPolicy UP>
struct SolverBoxSelector{
    using type = void;
};

template<typename T, size_t N>
struct SolverBoxSelector<T, N, UtilPolicy::RichVirtual>{
    using type = pbox::Box<OdeRichSolver<T, N>>;
};

template<typename T, size_t N>
struct SolverBoxSelector<T, N, UtilPolicy::Virtual>{
    using type = pbox::Box<OdeSolver<T, N>>;
};

} // namespace ode::detail


namespace traits{

template<typename T, size_t N, SolverPolicy SP>
struct HelperVirtualSolver{ using type = EmptySolver;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::Virtual>{ using type = OdeSolver<T, N>;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::RichVirtual>{ using type = OdeRichSolver<T, N>;};

template<typename T, size_t N, SolverPolicy SP>
using BaseInterface = typename HelperVirtualSolver<T, N, SP>::type;


template<typename Solver, typename T, size_t N, SolverPolicy SP>
using SolverCloneType = std::conditional_t<SP==SolverPolicy::Virtual || SP==SolverPolicy::RichVirtual, OdeSolver<T, N>, Solver>;


template<SolverPolicy SP>
constexpr bool is_rich = (SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual);

/// @brief Traits struct mapping a solver type to its virtual base interface.
/// Specialize this before the solver class definition to override the default.
template<typename Derived, typename T, size_t N, SolverPolicy SP>
struct SolverVirtualTypeTraits {
    using type = BaseInterface<T, N, SP>;
};

} // namespace ode::traits

} // namespace ode

#endif // ODEPACK_VIRTUAL_TRAITS_HPP