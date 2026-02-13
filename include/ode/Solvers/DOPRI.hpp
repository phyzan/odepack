#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

#include "../Core/RichBase.hpp"

namespace ode {

// Forward declarations
template<typename T, size_t Nstages>
T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size);

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
class RungeKuttaBase : public BaseDispatcher<Derived, T, N, SP, RhsType, JacType>{

    using Base = BaseDispatcher<Derived, T, N, SP, RhsType, JacType>;
protected:

    // ================ STATIC OVERRIDES ========================
    static constexpr bool   IS_IMPLICIT = false;

    StepResult  adapt_impl(T* res);

    void        reset_impl();

    void        re_adjust_impl(const T* new_vector);

    // =========================================================

    using RKBase = RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>;

    using Atype = Array2D<T, Nstages, Nstages>;
    using Btype = Array1D<T, Nstages>;
    using Ctype = Array1D<T, Nstages>;

    Atype A = Derived::Amatrix();
    Btype B = Derived::Bmatrix();
    Ctype C = Derived::Cmatrix();

    RungeKuttaBase(SOLVER_CONSTRUCTOR(T), size_t Krows) requires (!is_rich<SP>);

    RungeKuttaBase(SOLVER_CONSTRUCTOR(T), EVENTS events, size_t Krows) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RungeKuttaBase)

     void set_coef_matrix() const;

    void        step_impl(T* result, const T& h);

    // ======================= OVERRIDE =======================
     T    estimate_error_norm(const T* K, const T* scale, T h) const;

     void set_coef_matrix_impl() const;

    // ========================================================

    mutable Array2D<T, 0, N>    _K_true;
    mutable Array1D<T, N>       _df_tmp;
    mutable Array1D<T, N>       _scale_tmp;
    mutable Array1D<T, N>       _error_tmp;
    mutable Array2D<T, N, 0>    _coef_mat;
    mutable bool                _mat_is_set = false;
    T                           ERR_EXP = T(-1)/T(Derived::ERR_EST_ORDER+1);
};


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
class StandardRungeKutta : public RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>{

    using Base = RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>;

    friend Base;

protected:

    friend BaseSolver<Derived, T, N, SP, RhsType, JacType>; // So that Base can access specific private methods for static override

    using Etype = Array1D<T, Nstages+1>;
    using Ptype = Array2D<T, Nstages+1, 0>;

    Etype E = Derived::Ematrix();
    Ptype P = Derived::Pmatrix();

    StandardRungeKutta(SOLVER_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    StandardRungeKutta(SOLVER_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(StandardRungeKutta)

    // ================ STATIC OVERRIDES ========================

     std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

     void                                 interp_impl(T* result, const T& t) const;

    void                                        set_coef_matrix_impl() const;

     T                                    estimate_error_norm(const T* K, const T* scale, T h) const;

    // ==========================================================
};



template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class RK45 : public StandardRungeKutta<RK45<T, N, SP, RhsType, JacType>, T, N, 6, 5, SP, RhsType, JacType>{

public:

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK45);

private:

    using Base = StandardRungeKutta<RK45<T, N, SP, RhsType, JacType>, T, N, 6, 5, SP, RhsType, JacType>;
    friend Base;
    friend Base::RKBase;
    friend Base::MainSolverType;

    static const size_t Norder = 5;
    static const size_t Nstages = 6;

    static constexpr const char*    name = "RK45";
    static constexpr size_t         ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;

     static constexpr typename Base::Atype Amatrix();

     static constexpr typename Base::Btype Bmatrix();

     static constexpr typename Base::Ctype Cmatrix();

     static constexpr typename Base::Etype Ematrix();

     static constexpr typename Base::Ptype Pmatrix();

};





template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class RK23 : public StandardRungeKutta<RK23<T, N, SP, RhsType, JacType>, T, N, 3, 3, SP, RhsType, JacType> {

public:

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK23);

private:

    using Base = StandardRungeKutta<RK23<T, N, SP, RhsType, JacType>, T, N, 3, 3, SP, RhsType, JacType>;
    friend Base;
    friend Base::RKBase;
    friend Base::MainSolverType;

    static const size_t Norder = 3;
    static const size_t Nstages = 3;

    static constexpr const char*    name = "RK23";
    static constexpr size_t         ERR_EST_ORDER = 2;
    static constexpr size_t         INTERP_ORDER = 3;

     static constexpr typename Base::Atype Amatrix();

     static constexpr typename Base::Btype Bmatrix();

     static constexpr typename Base::Ctype Cmatrix();

     static constexpr typename Base::Etype Ematrix();

     static constexpr typename Base::Ptype Pmatrix();

};


} // namespace ode

#endif
