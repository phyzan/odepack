#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP

#include "../Core/Events.hpp"
#include "../OdeInt.hpp"
#include "../Core/VirtualBase.hpp"
#include "odepack/ode/IntegratorEnum.hpp"
#include "polybox/polybox.hpp"

namespace ode::chaos {

// ============================================================================
// DECLARATIONS
// ============================================================================


namespace detail{

// out (size 2*nsys) and in (size 2*nsys) can be the same pointer
template<typename T>
void normalized(T* out, const T* src, size_t nsys);

} // namespace ode::chaos::detail


template<typename T, size_t N, UtilPolicy UP>
class ChaoticSolver;

template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args);

template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class VariationalSolver;


template<typename T, size_t N, UtilPolicy UP>
class ChaoticSolver : public std::conditional_t<UP==UtilPolicy::RichVirtual, OdeRichSolver<T, 2*N>, OdeSolver<T, 2*N>>{

public:

    // ACCESSORS
    virtual void    get_rhs_main(T* out, const T& t, const T* q) const = 0;

    virtual void    get_jac_main(T* out, const T& t, const T* q, const T* dt = nullptr) const = 0;

    virtual T       get_elapsed_time() const = 0;

    virtual T       get_kick() const = 0;

    virtual T       get_period() const = 0;

    virtual T       get_log_ksi() const = 0;

    virtual T       get_lyapunov_exponent() const = 0;

    virtual T       get_stretching_number() const = 0;

};


template<typename T, size_t N, hasRhsFunc<T> OdeType>
struct VariationalOdeSys{

public:

    using DualType      = ::ode::DualType<T, N, 1>;
    using VarDualType   = ::ode::DualType<T, N, 2>;


    // same as in BaseSolver, but we need to redefine it here for the variational system
    static constexpr JacPolicy MAIN_JP = getJacPolicy<T, N, OdeType>();
    static constexpr JacPolicy JP = (getJacPolicy<T, 2*N, OdeType>() == JacPolicy::Autodiff) ? JacPolicy::Autodiff : JacPolicy::Approx;
    static_assert(MAIN_JP != JacPolicy::Approx,  "VariationalSolver requires the base solver to have an exact jacobian for the original system");


    VariationalOdeSys(OdeType ode, size_t ode_nsys);

    void    Rhs(T* out, const T& t, const T* q, const T* args) const;

    // Only provided if it does not require finite differences, otherwise the base solver will automatically use jac_approx to compute the jacobian of the full system.
    void    Jac(T* out, const T& t, const T* q, const T* args, const T* dt = nullptr) const requires (JP == JacPolicy::Autodiff);

    const OdeType& ode() const;

private:
    OdeType ode_;
    mutable Array1D<DualType, 2*N> diff_worker;
    mutable Array1D<VarDualType, 2*N> jac_worker;
    mutable Array2D<T, N, N, ndspan::Allocation::Auto, ndspan::Layout::F> jm;
    size_t nsys = N; // Size of the original system, without the variational equations (not the augmented system)
};


template<Integrator Solver, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
class VariationalSolver : public SolverTypeGetter<Solver, T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>>::type {

    /**
    This solvers integrates the coupled / augmented ODE system consisting of the original system and the variational equations. The state vector is of size 2*N, where the first N entries correspond to the state of the original system and the last N entries correspond to the deviation vector. The deviation vector is renormalized every "period" time units, and log_ksi keeps track of the logarithm of the stretching factor since the last renormalization. The user can retrieve the current value of the largest Lyapunov exponent using lyapunov_exponent(), which is computed as log_ksi divided by the elapsed time since the beginning of the integration.

    The augmented system is:

    dq/dt = f(q, t)
    d(delta_q)/dt = J(q, t) * delta_q

    The Jacobian of the original system is required to construct the variational equations.
    For implicit solvers, the Jacobian of the full system (original + variational) is also required.
    If the solver is provided with a templated Rhs function, then all jacobians are computed using autodiff. Otherwise, the solver must provide an exact jacobian for the original system, and finite differences are used to compute the jacobian of the full system, which schematically looks like:

    [           J           0  ]
    [ d(J)/dq *delta_q      J  ]

    */

public:

    using Base = typename SolverTypeGetter<Solver, T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<Solver, T, N, SP, OdeType, Derived>, Derived>>::type;

    template<typename... Args>
    VariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args = {}, Args&&... extra);

    T elapsed_time() const;

    T stretching_number() const;

    T kick() const;

    T period() const;

    T log_ksi() const;

    T lyapunov_exponent() const;

    void    Reset();

    void    RhsMain(T* out, const T& t, const T* q) const;

    void    JacMain(T* out, const T& t, const T* q, const T* dt = nullptr) const;

    // VIRTUAL INTERFACE ALIASES (inline overrides to avoid virtual calls)
    void    get_rhs_main(T* out, const T& t, const T* q) const { RhsMain(out, t, q); }
    void    get_jac_main(T* out, const T& t, const T* q, const T* dt = nullptr) const { JacMain(out, t, q, dt); }
    T       get_elapsed_time() const { return elapsed_time(); }
    T       get_kick() const { return kick(); }
    T       get_period() const { return period(); }
    T       get_log_ksi() const { return log_ksi(); }
    T       get_lyapunov_exponent() const { return lyapunov_exponent(); }
    T       get_stretching_number() const { return stretching_number(); }

protected:

    void    ReAdjust(const T* new_vector);

    template<typename... Args>
    bool Adv_Impl(Args&&... args);

private:
    struct private_tag{};

    static Array1D<T, 2*N> join_arrays(View1D<T, N> q0, View1D<T, N> delta_q0);

    mutable Array1D<T, 4*N> worker;
    Array1D<T, 2*N> tmp_state_;
    T period_;
    T t_next_; // the next time at which to renormalize
    T t_last_;
    T logksi_ = 0; // log of the norm of the deviation vector at the last renormalization, used for computing the Lyapunov exponent
    T logksi_last_ = 0; // log of the norm of the deviation vector at the previous renormalization, used for computing the Lyapunov exponent
    size_t np = 0;
    bool flagged = false;
};





template<typename T, size_t N>
class VariationalODE : public ODE<T, N>{

public:

    using Base = ODE<T, N>;

    template<hasRhsFunc<T> OdeType>
    VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args = {}, EventList<T> events = {}, Integrator method = Integrator::RK45);

    std::unique_ptr<ODE<T, N>> clone() const override;

    const std::vector<T>& renorm_times() const;

    const std::vector<T>& lyap_values() const;

    const std::vector<T>& kick_values() const;

    DEFAULT_RULE_OF_FOUR(VariationalODE)

    void clear() override;

    void reset() override;

    const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* solver() const;

protected:

    void register_state() override;

private:

    std::vector<T> renorm_times_ = {};
    std::vector<T> lyap_values_ = {};
    std::vector<T> kick_values_ = {};

};



template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args);


template<SolverPolicy SP, Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!traits::is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={});


template<SolverPolicy SP, Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (traits::is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, std::vector<T> args={}, EventList<T> events = {});

} // namespace ode::chaos


namespace ode{

template<Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct traits::SolverVirtualTypeTraits<chaos::VariationalSolver<Solver, T, N, SolverPolicy::Virtual, OdeType, DerivedVS>, T, NBase, SolverPolicy::Virtual> {
    using type = chaos::ChaoticSolver<T, N, UtilPolicy::Virtual>;
};


template<Integrator Solver, typename T, size_t N, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct traits::SolverVirtualTypeTraits<chaos::VariationalSolver<Solver, T, N, SolverPolicy::RichVirtual, OdeType, DerivedVS>, T, NBase, SolverPolicy::RichVirtual> {
    using type = chaos::ChaoticSolver<T, N, UtilPolicy::RichVirtual>;
};

} // namespace ode

#endif
