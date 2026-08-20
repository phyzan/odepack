#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

#include "../Core/RichBase.hpp"
#include "odepack/ode/IntegratorEnum.hpp"

namespace ode {

namespace detail{

template<typename T, size_t NSYS, size_t NCOUNT> class StaticRKScratch;
template<typename T, size_t NSYS, size_t NCOUNT> class DynamicRKScratch;

// Same rule as the solver's own scratch (see scratch_is_static in SolverBase.hpp): automatic
// storage only for a compile-time size and a trivial scalar; everything else is heap-backed
// and allocated once.
template<typename T, size_t NSYS, size_t NCOUNT>
using RKScratchSpace = std::conditional_t<scratch_is_static<T, NSYS>,
                                          StaticRKScratch<T, NSYS, NCOUNT>,
                                          DynamicRKScratch<T, NSYS, NCOUNT>>;

// A Butcher-tableau constant (A, B, C, E, P, ...) as a class member: a static constexpr
// table (shared, zero per-instance storage) when T is arithmetic, since it is then a
// compile-time constant; otherwise (e.g. T = mpfr::mpreal, which cannot be constant-evaluated)
// a plain instance member computed once at construction. Either way it's used the same:
// Coef(1,0), Coef[i], Coef.data().
template<typename T, auto Generator>
struct StaticCoefTable{
    using Matrix = decltype(Generator());
    static constexpr Matrix table = Generator();

    template<typename... Idx>
    inline constexpr const auto& operator()(Idx... idx) const { return table(idx...); }
    inline constexpr const auto& operator[](size_t i) const { return table[i]; }
    inline constexpr const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
struct DynamicCoefTable{
    using Matrix = decltype(Generator());
    Matrix table = Generator();

    template<typename... Idx>
    inline const auto& operator()(Idx... idx) const { return table(idx...); }
    inline const auto& operator[](size_t i) const { return table[i]; }
    inline const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
using CoefTable = std::conditional_t<std::is_arithmetic_v<T>, StaticCoefTable<T, Generator>, DynamicCoefTable<T, Generator>>;



// ============================================================================
// Shared explicit Runge-Kutta building blocks, used by RK23, RK45 and DOP853.
// The per-stage arithmetic itself (e.g. h * (a21*K0 + ...)) stays hardcoded in each
// solver's step_impl for performance; only the surrounding, method-agnostic machinery
// (step-size control, dense-output coefficient assembly) is shared here.
// ============================================================================

/// @brief Build the dense-output polynomial coefficient matrix (n x order) from stage
/// derivatives K (Nstages+1 rows) and interpolation weights P (Nstages+1 x order).
template<typename T>
void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n);

/// @brief One Dormand-Prince 5(4) stage sweep, free of any solver state. Writes t+h and the
/// new state into `result` ([0] = t+h, [2..] = q_new), stages K1..K5 into `K` (Nstages-1 rows
/// of n) and the final FSAL stage into `KF`; `K0` is the derivative at the start of the step
/// and `r` is scratch of length n. Returns the scaled error norm. It lives outside the class
/// so the dense-output coefficients can replay a finished step's stages without duplicating
/// the tableau arithmetic.
// Whether the stage sweep below should be forced into its caller. The two compilers want
// opposite things here, and the difference is large, so it is measured rather than guessed
// (harmonic oscillator, N=2, 3.06M steps, -O3 -march=native):
//
//                       out-of-line     always_inline
//        gcc 13.3          74.5 ms         141.4 ms
//        clang 18.1       147.5 ms         104.4 ms
//
// gcc keeps the call site tight and re-derives the constants across it; forcing the sweep in
// bloats the retry loop and it spills. clang does not propagate into the out-of-line call and
// only gets there by inlining. Define ODEPACK_FORCE_INLINE_STEP=0/1 to override.
#ifndef ODEPACK_FORCE_INLINE_STEP
    #if defined(__clang__)
        #define ODEPACK_FORCE_INLINE_STEP 1
    #else
        #define ODEPACK_FORCE_INLINE_STEP 0
    #endif
#endif

#if ODEPACK_FORCE_INLINE_STEP
    #define ODEPACK_STEP_ATTR [[gnu::always_inline]]
#else
    #define ODEPACK_STEP_ATTR
#endif

/// NSYS is the solver's compile-time system size (0 when it is only known at run time).
/// It is a template parameter rather than just the `nsys` argument so that the stage loops
/// keep a constant trip count even when the compiler chooses not to inline this function -
/// without it, a compiler that leaves the call out of line loses the size and emits full
/// vector + epilogue paths for all seven loops.
template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
ODEPACK_STEP_ATTR T rk45_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

/// @brief One Bogacki-Shampine 3(2) stage sweep. Same contract as rk45_step_impl: `K` holds
/// stages K1..K2 (Nstages-1 rows of n), `KF` the final FSAL stage, `K0` the derivative at the
/// start of the step. Returns the scaled error norm.
template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
ODEPACK_STEP_ATTR T rk23_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

/// @brief Shared step-size control loop: repeatedly calls step_fn(res, state, h) -> err_norm,
/// halving/growing habs until the local error is accepted (mirrors scipy/boost step control).
template<typename T, typename StepFn>
StepResult rk_adapt_step(T* res, const T* state, size_t n,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn);

} // namespace ode::detail

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK23 : public detail::BaseDispatcher<GetDerived<RK23<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK23<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t Nstages       = 3;
    static constexpr size_t Norder        = 3;
    static constexpr size_t INTERP_ORDER  = 3;
    static constexpr int    ERR_EST_ORDER = 2;
    static constexpr bool   IS_IMPLICIT   = false;

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>);

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T), EventList<T> events = {}) requires (traits::is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK23)

    Integrator  method() const;

    auto        local_interp() const;

    void        Reset();

protected:

    void        ReAdjust(const T* new_vector);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

private:

    using Atype = Array2D<T, Nstages, Nstages, Allocation::Stack>;
    using Btype = Array1D<T, Nstages, Allocation::Stack>;
    using Ctype = Array1D<T, Nstages, Allocation::Stack>;
    using Etype = Array1D<T, Nstages+1, Allocation::Stack>;
    using Ptype = Array2D<T, Nstages+1, INTERP_ORDER, Allocation::Stack>;

    static constexpr Atype Amatrix();
    static constexpr Btype Bmatrix();
    static constexpr Ctype Cmatrix();
    static constexpr Etype Ematrix();
    static constexpr Ptype Pmatrix();

    T           step_impl(T* result, const T* state, const T& h);

    void        set_coef_matrix() const;

    detail::CoefTable<T, &Amatrix> A;
    detail::CoefTable<T, &Bmatrix> B;
    detail::CoefTable<T, &Cmatrix> C;
    detail::CoefTable<T, &Ematrix> E;
    detail::CoefTable<T, &Pmatrix> P;

    detail::RKScratchSpace<T, N, Nstages-1>              scratch_space;
    T                                                   h_last_ = 0; // replayed by set_coef_matrix
    mutable Array1D<T, N>                               K0_;  // derivative at the start of the step
    mutable Array1D<T, N>                               KF_;  // final (FSAL) stage
    mutable Array2D<T, N, 0>                            coef_mat_;
    mutable bool                                        mat_is_set_ = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1); // Boost uses -1/(error_order+1) for both increase and decrease
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK45 : public detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t Nstages       = 6;
    static constexpr size_t Norder        = 5;
    static constexpr size_t INTERP_ORDER  = 4;
    static constexpr int    ERR_EST_ORDER = 4;
    static constexpr bool   IS_IMPLICIT   = false;

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>);

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T), EventList<T> events = {}) requires (traits::is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK45)

    Integrator method() const;

    auto local_interp() const;

    void        Reset();

protected:

    void        ReAdjust(const T* new_vector);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

private:

    using Atype = Array2D<T, Nstages, Nstages, Allocation::Stack>;
    using Btype = Array1D<T, Nstages, Allocation::Stack>;
    using Ctype = Array1D<T, Nstages, Allocation::Stack>;
    using Etype = Array1D<T, Nstages+1, Allocation::Stack>;
    using Ptype = Array2D<T, Nstages+1, INTERP_ORDER, Allocation::Stack>;

    static constexpr Atype Amatrix();
    static constexpr Btype Bmatrix();
    static constexpr Ctype Cmatrix();
    static constexpr Etype Ematrix();
    static constexpr Ptype Pmatrix();

    T           step_impl(T* result, const T* state, const T& h);

    void        set_coef_matrix() const;

    detail::CoefTable<T, &Amatrix> A;
    detail::CoefTable<T, &Bmatrix> B;
    detail::CoefTable<T, &Cmatrix> C;
    detail::CoefTable<T, &Ematrix> E;
    detail::CoefTable<T, &Pmatrix> P;

    detail::RKScratchSpace<T, N, Nstages-1> scratch_space;
    T                                       h_last_ = 0; // step size of the last sweep, replayed by set_coef_matrix
    mutable Array1D<T, N>                   K0_;
    mutable Array1D<T, N>                   KF_;
    mutable Array2D<T, N, 0>                coef_mat;
    mutable bool                            mat_is_set = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1); // Boost uses -1/(error_order+1) for both increase and decrease
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::RK23, T, N, SP, OdeType, Derived>{
    using type = RK23<T, N, SP, OdeType, Derived>;
};


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::RK45, T, N, SP, OdeType, Derived>{
    using type = RK45<T, N, SP, OdeType, Derived>;
};

} // namespace ode

#endif
