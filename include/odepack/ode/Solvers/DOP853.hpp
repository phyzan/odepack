#ifndef DOP853_HPP
#define DOP853_HPP

#include "../Core/RichBase.hpp"


namespace ode{

// ============================================================================
// Raw Butcher-tableau constants for DOP853 (Hairer, Norsett & Wanner).
// Purely data - no solver logic lives here.
// ============================================================================

template<typename T>
struct DOP_COEFS{

    static constexpr size_t N_STAGES = 12;
    static constexpr size_t N_STAGES_EXT = 16;
    static constexpr size_t INTERP_ORDER = 7;
    static constexpr int ERR_EST_ORDER = 7;

    using DOP_A = Array2D<T, N_STAGES_EXT, N_STAGES_EXT, Allocation::Stack>;
    using DOP_B = Array1D<T, N_STAGES, Allocation::Stack>;
    using DOP_C = Array1D<T, N_STAGES_EXT, Allocation::Stack>;
    using DOP_D = Array2D<T, INTERP_ORDER - 3, N_STAGES_EXT, Allocation::Stack>;
    using DOP_E = Array1D<T, N_STAGES+1, Allocation::Stack>;

    static constexpr DOP_A make_A();

    static constexpr DOP_B make_B();

    static constexpr DOP_C make_C();

    static constexpr DOP_E make_E3();

    static constexpr DOP_E make_E5();

    static constexpr DOP_D make_D();

};

template<typename T>
void coef_mat_interp_dop853(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size);

/// @brief DOP853's combined 3rd/5th order embedded error norm (Hairer, Norsett & Wanner).
template<typename T>
T dop853_error_norm(const T* K, const T* E3, const T* E5, const T* q, const T* q_new,
                     const T& h, const T& rtol, const T& atol, size_t Nstages, size_t n);


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class DOP853 : public detail::BaseDispatcher<GetDerived<DOP853<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<DOP853<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t N_STAGES       = DOP_COEFS<T>::N_STAGES;       // 12
    static constexpr size_t N_STAGES_EXTRA = 3;                            // extra stages, dense output only
    static constexpr size_t N_STAGES_EXT   = DOP_COEFS<T>::N_STAGES_EXT;   // 16
    static constexpr size_t Norder         = 8;
    static constexpr size_t INTERP_ORDER   = DOP_COEFS<T>::INTERP_ORDER;   // 7
    static constexpr int    ERR_EST_ORDER  = DOP_COEFS<T>::ERR_EST_ORDER;  // 7
    static constexpr bool   IS_IMPLICIT    = false;

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>);

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T), EventList<T> events = {}) requires (traits::is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(DOP853)

    Integrator method() const;

    auto local_interp() const;

    void Reset();

protected:

    void        ReAdjust(const T* new_vector);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

private:

    using Atype      = Array2D<T, N_STAGES, N_STAGES, Allocation::Stack>;
    using Btype      = Array1D<T, N_STAGES, Allocation::Stack>;
    using Ctype      = Array1D<T, N_STAGES, Allocation::Stack>;
    using AExtraType = Array2D<T, N_STAGES_EXTRA, N_STAGES_EXT, Allocation::Stack>;
    using CExtraType = Array1D<T, N_STAGES_EXTRA, Allocation::Stack>;

    static constexpr Atype      Amatrix();
    static constexpr Btype      Bmatrix();
    static constexpr Ctype      Cmatrix();
    static constexpr AExtraType Amatrix_extra();
    static constexpr CExtraType Cmatrix_extra();

    T           step_impl(T* result, const T* state, const T& h);

    void        set_coef_matrix() const;

    // Unlike RK23/RK45, these tables are built with fill()+indexed writes rather than a
    // plain aggregate initializer, which ndspan cannot constant-evaluate - so, regardless
    // of T, they are always plain (computed-once-at-construction) instance members here.
    Atype      A       = Amatrix();
    Btype      B       = Bmatrix();
    Ctype      C       = Cmatrix();
    AExtraType A_extra = Amatrix_extra();
    CExtraType C_extra = Cmatrix_extra();
    typename DOP_COEFS<T>::DOP_D D  = DOP_COEFS<T>::make_D();
    typename DOP_COEFS<T>::DOP_E E3 = DOP_COEFS<T>::make_E3();
    typename DOP_COEFS<T>::DOP_E E5 = DOP_COEFS<T>::make_E5();

    T                                                     h_last_ = 0; // step size actually used by the last sweep
    mutable Array2D<T, N_STAGES_EXT, N, Allocation::Auto> K_;
    mutable Array1D<T, N, Allocation::Auto>                df_tmp_;
    mutable Array2D<T, N, 0>                               coef_mat_;
    mutable bool                                           mat_is_set_ = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1);
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::DOP853, T, N, SP, OdeType, Derived>{
    using type = DOP853<T, N, SP, OdeType, Derived>;
};


} // namespace ode

#endif
