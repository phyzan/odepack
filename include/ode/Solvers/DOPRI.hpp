#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

#include "../Core/RichBase.hpp"

namespace ode {

// Forward declarations
// template<typename T, size_t Nstages>
// T error_norm(const T* E, const T* K, const T* q, const T* q_new, const T& rtol, const T& atol, const T& h, size_t size);

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
class RungeKuttaMainBase : public BaseDispatcher<Derived, T, N, SP, RhsType, JacType>{

    
protected:

    // ================ STATIC OVERRIDES ========================
    static constexpr bool   IS_IMPLICIT = false;

    T    step_impl(T* result, const T* state, const T& h);

    StepResult  adapt_impl(T* res, const T* state);

    void        reset_impl();

    void        re_adjust_impl(const T* new_vector);

    // =========================================================
    using Base = BaseDispatcher<Derived, T, N, SP, RhsType, JacType>;
    using RKBase = RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>;

    using Atype = Array2D<T, Nstages, Nstages, Allocation::Stack>;
    using Btype = Array1D<T, Nstages, Allocation::Stack>;
    using Ctype = Array1D<T, Nstages, Allocation::Stack>;

    RungeKuttaMainBase(SOLVER_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RungeKuttaMainBase(SOLVER_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RungeKuttaMainBase)

    void set_coef_matrix() const;

    // ======================= OVERRIDE =======================

    void set_coef_matrix_impl() const;

    // ========================================================

    mutable Array2D<T, K_ROWS, N, Allocation::Auto>    K_;
    mutable Array1D<T, N, Allocation::Auto>       _df_tmp;
    mutable Array2D<T, N, 0>    _coef_mat;
    mutable bool                _mat_is_set = false;
    T                           ERR_EXP = T(-1)/T(Derived::ERR_EST_ORDER + 1); // Boost uses -1/(error_order+1) for both increase and decrease
    T                           INC_EXP = T(-1) / Norder;
    T                           MIN_ERR = T(1) / pow(T(5), Norder);
};


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
class RungeKuttaBaseStatic : public RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>{

protected:
    using Base = RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>;
    using Base::Base;

    static constexpr typename Base::Atype Am_ = Derived::Amatrix();
    static constexpr typename Base::Btype Bm_ = Derived::Bmatrix();
    static constexpr typename Base::Ctype Cm_ = Derived::Cmatrix();
};

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
class RungeKuttaBaseDynamic : public RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>{

protected:
    using Base = RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>;
    using Base::Base;

    typename Base::Atype Am_ = Derived::Amatrix();
    typename Base::Btype Bm_ = Derived::Bmatrix();
    typename Base::Ctype Cm_ = Derived::Cmatrix();
};

template<typename Derived, typename T, size_t N, size_t Nstates, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
using DOPRI_DISPATCHER = std::conditional_t<std::is_arithmetic_v<T>, RungeKuttaBaseStatic<Derived, T, N, Nstates, Norder, K_ROWS, SP, RhsType, JacType>, RungeKuttaBaseDynamic<Derived, T, N, Nstates, Norder, K_ROWS, SP, RhsType, JacType>>;


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
class StandardRungeKuttaBase : public DOPRI_DISPATCHER<Derived, T, N, Nstages, Norder, Nstages+1, SP, RhsType, JacType>{

protected:
    using Base = DOPRI_DISPATCHER<Derived, T, N, Nstages, Norder, Nstages+1, SP, RhsType, JacType>;
    friend Base;
    friend Base::Base;
    friend BaseSolver<Derived, T, N, SP, RhsType, JacType>; // So that Base can access specific private methods for static override

    using Etype = Array1D<T, Nstages+1, Allocation::Stack>;

    StandardRungeKuttaBase(SOLVER_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    StandardRungeKuttaBase(SOLVER_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(StandardRungeKuttaBase)
    // ================ STATIC OVERRIDES ========================

    std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

    void                                 interp_impl(T* result, const T& t) const;

    void                                 set_coef_matrix_impl() const;

    // ==========================================================

    T estimate_error_norm(const T* K, const T* q, const T* q_new, const T& rtol, const T& atol, const T& h) const;

};

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
class StandardRungeKuttaBaseStatic : public StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>{

    
protected:
    using Base = StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>;
    using Base::Base;
    friend Base;
    friend Base::Base;
    friend Base::Base::Base;

    static constexpr typename Base::Etype Em_ = Derived::Ematrix();

};

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
class StandardRungeKuttaBaseDynamic : public StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>{


protected:
    using Base = StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>;
    using Base::Base;
    friend Base;
    friend Base::Base;
    friend Base::Base::Base;

    typename Base::Etype Em_ = Derived::Ematrix();

};

template<typename Derived, typename T, size_t N, size_t Nstates, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
using STANDARD_DOPRI_DISPATCHER = std::conditional_t<std::is_arithmetic_v<T>, StandardRungeKuttaBaseStatic<Derived, T, N, Nstates, Norder, SP, RhsType, JacType>, StandardRungeKuttaBaseDynamic<Derived, T, N, Nstates, Norder, SP, RhsType, JacType>>;


template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>, typename Derived = void>
class RK45 : public STANDARD_DOPRI_DISPATCHER<GetDerived<RK45<T, N, SP, RhsType, JacType, Derived>, Derived>, T, N, 6, 5, SP, RhsType, JacType>{

public:

    static constexpr size_t Norder = 5;
    static constexpr size_t Nstages = 6;

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK45);

protected:

    using Base = STANDARD_DOPRI_DISPATCHER<GetDerived<RK45<T, N, SP, RhsType, JacType, Derived>, Derived>, T, N, 6, 5, SP, RhsType, JacType>;
    using Ptype = Array2D<T, Nstages+1, 4, Allocation::Stack>;
    
    friend Base;
    friend Base::Base;
    friend Base::Base::Base;
    friend Base::Base::Base::Base;
    friend Base::MainSolverType;

    T    step_impl(T* result, const T* state, const T& h);

    static constexpr const char*    name = "RK45";
    static constexpr size_t         ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;

    static constexpr typename Base::Atype Amatrix();

    static constexpr typename Base::Btype Bmatrix();

    static constexpr typename Base::Ctype Cmatrix();

    static constexpr typename Base::Etype Ematrix();

    static constexpr                Ptype Pmatrix();

private:
    Ptype Pm_ = Pmatrix();

};





template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>, typename Derived = void>
class RK23 : public STANDARD_DOPRI_DISPATCHER<GetDerived<RK23<T, N, SP, RhsType, JacType, Derived>, Derived>, T, N, 3, 3, SP, RhsType, JacType> {

public:

    static constexpr size_t Norder = 3;
    static constexpr size_t Nstages = 3;

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK23);

protected:

    using Base = STANDARD_DOPRI_DISPATCHER<GetDerived<RK23<T, N, SP, RhsType, JacType, Derived>, Derived>, T, N, 3, 3, SP, RhsType, JacType>;
    using Ptype = Array2D<T, Nstages+1, 3, Allocation::Stack>;
    friend Base;
    friend Base::Base;
    friend Base::Base::Base;
    friend Base::Base::Base::Base;
    friend Base::MainSolverType;

    T    step_impl(T* result, const T* state, const T& h);

    static constexpr const char*    name = "RK23";
    static constexpr size_t         ERR_EST_ORDER = 2;
    static constexpr size_t         INTERP_ORDER = 3;

    static constexpr typename Base::Atype Amatrix();

    static constexpr typename Base::Btype Bmatrix();

    static constexpr typename Base::Ctype Cmatrix();

    static constexpr typename Base::Etype Ematrix();

    static constexpr                Ptype Pmatrix();

private:
    Ptype Pm_ = Pmatrix();

};


} // namespace ode

#endif
