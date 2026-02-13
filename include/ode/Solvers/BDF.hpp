#ifndef BDF_HPP
#define BDF_HPP

#include "../Core/RichBase.hpp"

namespace ode{

constexpr size_t BDF_MAX_ORDER = 5;

// ============================================================================
// DECLARATIONS
// ============================================================================

// Struct declarations
template<typename T, size_t N>
struct LUResult {
    LUResult(size_t Nsys);
    JacMat<T, N> LU;
    Array1D<size_t, N> piv;

    void lu_factor(const JacMat<T, N>& A_input);

    void lu_solve(T* x, const T* b) const;
};

template<typename T>
struct BDFCONSTS{
    Array1D<T, BDF_MAX_ORDER+1> KAPPA;
    Array1D<T, BDF_MAX_ORDER+1> GAMMA;
    Array1D<T, BDF_MAX_ORDER+1> ALPHA;
    Array1D<T, BDF_MAX_ORDER+1> ERR_CONST;

    BDFCONSTS();
};

struct NewtConv{
    bool converged;
    size_t n_iter;
    StepResult flag;
};

// Function declarations

template<typename T>
Array1D<T> arange(size_t a, size_t b);

template<typename T>
void cumprod(T* res, const T* x, size_t size);

template<typename T>
void compute_R(T* R, size_t order, T factor);

template<typename T>
void bdf_interp(T* result, const T& t, const T& t2, const T& h, const T* D, size_t order, size_t size);

// Class declarations
template<typename T, size_t N>
class BDFInterpolator final : public LocalInterpolator<T, N>{

public:

    BDFInterpolator() = delete;

    DEFAULT_RULE_OF_FOUR(BDFInterpolator);

    BDFInterpolator(const T& t, const T* q, size_t nsys);

    BDFInterpolator(const Array2D<T, 0, N>& D, size_t order, const T* state1, const T* state2, size_t nsys, int bdr1, int bdr2);

    size_t                  order() const final;

    BDFInterpolator<T, N>*  clone() const final;

private:

    void _call_impl(T* result, const T& t) const final;

    size_t _order = 1;
    T _h;
    T _t2;
    Array2D<T, 0, N> _D;

};


template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class BDF : public BaseDispatcher<BDF<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>{

public:

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>) : BDF(ARGS, None()) {}

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>) : BDF(ARGS, None(), events) {}

     std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

    DEFAULT_RULE_OF_FOUR(BDF)

private:

    using Base = BaseDispatcher<BDF<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>;
    using Dlike = Array2D<T, 0, N>;
    struct None{};
    friend Base::MainSolverType;
    static constexpr size_t NEWTON_MAXITER = 4;

    static constexpr const char* name = "BDF";
    static constexpr bool IS_IMPLICIT = true;
    static constexpr int ERR_EST_ORDER = 1;

    StepResult      adapt_impl(T* res);
    void            interp_impl(T* result, const T& t) const;
    void            reset_impl();
    void            re_adjust_impl(const T* new_vector);
    bool            validate_ics_impl(T t0, const T* q0) const;

    template<typename... Type>
    BDF(MAIN_CONSTRUCTOR(T), None, Type&&... extras);

    void    _reset_impl_alone();

    NewtConv _solve_bdf_system(T* y, const T* y_pred, Array1D<T, N>& d, const T& t_new, const T& c, const Array1D<T, N>& psi, const LUResult<T, N>& LU, const Array1D<T, N>& scale);

    void _change_D(const T& factor);

    void _set_prediction(T* y);

    void _set_psi(T* psi);

    bool _resize_step(T& factor, const T& min_step, const T& max_step);

    JacMat<T, N> _J;
    mutable JacMat<T, N> _B;
    std::array<Dlike, 3> _D;
    LUResult<T, N> _LU;
    T _newton_tol;
    size_t _order = 1;
    size_t _n_eq_steps = 0;
    bool _valid_LU = false;
    size_t _idx_D = 0;
    mutable std::vector<T> _R, _U, _RU;
    mutable Array1D<T, N> _f, _dy, _b, _scale, _ypred, _psi, _d, _error, _error_m, _error_p;
    mutable std::array<T, 3> _error_norms;
    BDFCONSTS<T> BDF_COEFS;
    int interp_idx = 0;

};



} // namespace ode

#endif
