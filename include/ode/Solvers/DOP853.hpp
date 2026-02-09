#ifndef DOP853_HPP
#define DOP853_HPP

#include "DOPRI.hpp"

namespace ode{

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
struct DOP_COEFS{

     static constexpr size_t N_STAGES = 12;
     static constexpr size_t N_STAGES_EXT = 16;
     static constexpr size_t INTERP_ORDER = 7;
     static constexpr int ERR_EST_ORDER = 7;

    using DOP_A = Array2D<T, N_STAGES_EXT, N_STAGES_EXT>;
    using DOP_B = Array1D<T, N_STAGES>;
    using DOP_C = Array1D<T, 16>;
    using DOP_D = Array2D<T, INTERP_ORDER - 3, N_STAGES_EXT>;
    using DOP_E = Array1D<T, N_STAGES+1>;

    static DOP_A make_A();

    static DOP_B make_B();

    static DOP_C make_C();

    static DOP_E make_E3();

    static DOP_E make_E5();

    static DOP_D make_D();

    DOP_A A = make_A();
    DOP_B B = make_B();
    DOP_C C = make_C();
    DOP_E E3 = make_E3();
    DOP_E E5 = make_E5();
    DOP_D D = make_D();

};

template<typename T>
void coef_mat_interp_dop853(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size);


template<typename T, size_t N>
class DOP853LocalInterpolator final: public LocalInterpolator<T, N>{

public:

    DOP853LocalInterpolator() = delete;

    DOP853LocalInterpolator(const T& t, const T* q, size_t nsys);

    DOP853LocalInterpolator(const Array2D<T, N, 0>& coef_mat, T t1, T t2, const T* y1, const T* y2, size_t nsys, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(DOP853LocalInterpolator);

    size_t order() const override;

    DOP853LocalInterpolator<T, N>* clone() const override;

protected:

    Array2D<T, N, 0> _coef_mat;
    size_t _order = 0;

private:

    void _call_impl(T* result, const T& t) const override;

};


template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class DOP853 : public RungeKuttaBase<DOP853<T, N, SP, RhsType, JacType>, T, N, 12, 8, SP, RhsType, JacType>{

public:

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

     std::unique_ptr<Interpolator<T, N>> state_interpolator(int bdr1, int bdr2) const;

private:

    using Base = RungeKuttaBase<DOP853<T, N, SP, RhsType, JacType>, T, N, 12, 8, SP, RhsType, JacType>;
    friend Base::MainSolverType;

    static constexpr const char* name = "DOP853";
    
    static constexpr size_t N_STAGES = 12;
    static constexpr size_t N_ORDER = 8;
    static constexpr size_t N_STAGES_EXTRA = 3;
    static constexpr int    ERR_EST_ORDER = 7;
    static constexpr size_t N_STAGES_EXT = DOP_COEFS<T>::N_STAGES_EXT;
    static constexpr size_t INTERP_ORDER = DOP_COEFS<T>::INTERP_ORDER;
    

    

    using A_EXTRA_TYPE = Array2D<T, N_STAGES_EXTRA, N_STAGES_EXT>;

    using C_EXTRA_TYPE = Array1D<T, N_STAGES_EXTRA>;

    friend Base;
    friend Base::MainSolverType; // So that Base can access specific private methods for static override

     void interp_impl(T* result, const T& t) const;

    static typename Base::Atype Amatrix();

    static typename Base::Btype Bmatrix();

    static typename Base::Ctype Cmatrix();

    static A_EXTRA_TYPE Amatrix_extra();

    static C_EXTRA_TYPE Cmatrix_extra();

    A_EXTRA_TYPE A_EXTRA = Amatrix_extra();

    C_EXTRA_TYPE C_EXTRA = Cmatrix_extra();

    typename DOP_COEFS<T>::DOP_D D = DOP_COEFS<T>::make_D();

    typename DOP_COEFS<T>::DOP_E E3 = DOP_COEFS<T>::make_E3();

    typename DOP_COEFS<T>::DOP_E E5 = DOP_COEFS<T>::make_E5();

    void set_coef_matrix_impl() const;

    T estimate_error_norm(const T* K, const T* scale, T h) const;

};


} // namespace ode

#endif
