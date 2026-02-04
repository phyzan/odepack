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

    void adapt_impl(T* res);

    void reset_impl();

    void re_adjust_impl(const T* new_vector);

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

    inline void set_coef_matrix() const;

    void        step_impl(T* result, const T& h);

    // ======================= OVERRIDE =======================
    inline T    estimate_error_norm(const T* K, const T* scale, T h) const;

    inline void set_coef_matrix_impl() const;

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

    inline std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;

    inline void                                 interp_impl(T* result, const T& t) const;

    void                                        set_coef_matrix_impl() const;

    inline T                                    estimate_error_norm(const T* K, const T* scale, T h) const;

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

    inline static constexpr typename Base::Atype Amatrix();

    inline static constexpr typename Base::Btype Bmatrix();

    inline static constexpr typename Base::Ctype Cmatrix();

    inline static constexpr typename Base::Etype Ematrix();

    inline static constexpr typename Base::Ptype Pmatrix();

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

    inline static constexpr typename Base::Atype Amatrix();

    inline static constexpr typename Base::Btype Bmatrix();

    inline static constexpr typename Base::Ctype Cmatrix();

    inline static constexpr typename Base::Etype Ematrix();

    inline static constexpr typename Base::Ptype Pmatrix();

};



//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//--------------------------------IMPLEMENTATIONS-------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------




template<typename T>
inline void adapt_scale(T* scale, const T* q1, const T* q2, T atol, T rtol, size_t size){
    for (size_t i=0; i<size; i++){
        scale[i] = atol + std::max(abs(q1[i]), abs(q2[i]))*rtol;
    }
}

template<typename T, size_t Nstages>
T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size) {

    // std::fill(tmp, tmp+size, 0);
    // for (size_t i=0; i<Nstages+1; i++){
    //     T p = E[i]*h;
    //     #pragma omp simd
    //     for (size_t j=0; j<size; j++){
    //         tmp[j] += (K[i*size+j] * p)/scale[j];
    //     }
    // }

    for (size_t j=0; j<size; j++){
        T sum = 0;
        for (size_t i=0; i<Nstages+1; i++){
            sum += K[i*size+j] * E[i]*h;
        }
        tmp[j] = sum/scale[j];
    }
    return rms_norm(tmp, size);
}

// template<typename T, size_t Nstages>
// T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size) {
//     // Single-pass: compute error and max simultaneously
//     T max_err = 0;
//     for (size_t j = 0; j < size; j++) {
//         T sum = 0;
//         for (size_t i = 0; i < Nstages + 1; i++) {
//             sum += E[i] * K[i * size + j];
//         }
//         T err = abs(h * sum / scale[j]);
//         max_err = max(max_err, err);
//     }
//     return max_err;
// }


// RungeKuttaBase implementations
template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::RungeKuttaBase(SOLVER_CONSTRUCTOR(T), size_t Krows)
    requires (!is_rich<SP>): Base(ARGS), _K_true(Krows, nsys), _df_tmp(nsys), _scale_tmp(nsys), _error_tmp(nsys), _coef_mat(nsys, Derived::INTERP_ORDER) {
    // Compute K[Nstages] = f(t0, q0) for FSAL
    if (q0 != nullptr){
        this->rhs(_K_true.data()+Nstages*nsys, t0, q0);
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::RungeKuttaBase(SOLVER_CONSTRUCTOR(T), EVENTS events, size_t Krows)
    requires (is_rich<SP>): Base(ARGS, events), _K_true(Krows, nsys), _df_tmp(nsys), _scale_tmp(nsys), _error_tmp(nsys), _coef_mat(nsys, Derived::INTERP_ORDER) {
    // Compute K[Nstages] = f(t0, q0) for FSAL
    if (q0 != nullptr){
        this->rhs(_K_true.data()+Nstages*nsys, t0, q0);
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    _K_true.set(0);
    _mat_is_set = false;
    // Compute K[Nstages] = f(t0, q0) for FSAL
    const T* state = this->new_state_ptr();
    this->rhs(_K_true.data()+Nstages*this->Nsys(), state[0], state+2);
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
    // Re-compute K[Nstages] = f(t, q) after restarting at intermediate time (e.g. because of discontinuity due to masked event)
    Base::re_adjust_impl(new_vector);

    //prepate the coef matrix for interpolation
    //before altering K, so that interpolation is valid
    //from t_old to t
    this->set_coef_matrix();
    //copy the new state vector into the last stage of K
    this->rhs(_K_true.data() + Nstages*this->Nsys(), this->t(), new_vector);
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline T RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::estimate_error_norm(const T* K, const T* scale, T h) const {
    return THIS_C->estimate_error_norm(K, scale, h);
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::set_coef_matrix_impl() const{
    THIS_C->set_coef_matrix_impl();
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::set_coef_matrix() const{
    if (!this->_mat_is_set){
        this->set_coef_matrix_impl();
        this->_mat_is_set = true;
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::adapt_impl(T* res){
    this->_mat_is_set = false;
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const T& atol = this->atol();
    const T& rtol = this->rtol();
    const T* state = this->new_state_ptr();
    const size_t n = this->Nsys();

    T& habs = res[1];
    habs = state[1];
    T h, err_norm, factor, _factor;
    T* q_new = res+2;
    const T* q = state+2;

    bool step_accepted = false;
    bool step_rejected = false;
    T* K_ = _K_true.data();
    copy_array(K_, K_ + Nstages*n, n); //FSAL: K[0] for next step = K[Nstages] from this step. Doing this instead of calling this->rhs(t_current, q_current) in step_impl
    while (!step_accepted){
        h = habs * this->direction();
        step_impl(res, h); //res and K are altered
        adapt_scale(_scale_tmp.data(), q, q_new, atol, rtol, n);
        err_norm = this->estimate_error_norm(_K_true.data(), _scale_tmp.data(), h);
        _factor = this->SAFETY*pow(err_norm, ERR_EXP);
        if (err_norm < 1){
            factor = (err_norm == 0) ? this->MAX_FACTOR : std::min(this->MAX_FACTOR, _factor);
            if (step_rejected){
                factor = factor < 1 ? factor : 1;
            }
            step_accepted = true;
        }
        else{
            factor = std::max(this->MIN_FACTOR, _factor);
            step_rejected = true;
        }
        if (!resize_step(factor, habs, h_min, max_step)){
            break;
        }
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::step_impl(T* result, const T& h){
    const T* state = this->new_state_ptr();
    const size_t n = this->Nsys();
    T* q_new = result + 2;
    T* K_ = _K_true.data();
    const T* q = state + 2;
    const T* B_ = B.data();
    const T* C_ = C.data();
    const T* A_ = A.data();
    T* r = _df_tmp.data();
    const T& t = state[0];

    // Stage 1: K[0] = K[Nstages] (FSAL, already done in adapt_impl)

    // Initialize q_new = q + h*B[0]*K1
    const T hB0 = B_[0] * h;
    #pragma omp simd
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + hB0 * K_[j];
    }

    // Stages 2 through Nstages: fused single-pass computation
    for (size_t s = 1; s < Nstages; s++) {
        // Compute r = q + h * sum(A[s][i]*K[i]) in a single pass
        #pragma omp simd
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t i = 0; i < s; i++) {
                sum += A_[s*Nstages+i] * K_[i*n + j];
            }
            r[j] = q[j] + h * sum;
        }

        // K[s] = f(t + c[s]*h, r)
        this->rhs(K_ + s*n, t + C_[s]*h, r);

        // Accumulate into q_new
        const T hBs = B_[s] * h;
        #pragma omp simd
        for (size_t j = 0; j < n; j++) {
            q_new[j] += hBs * K_[s*n + j];
        }
    }

    // Final: K[Nstages] = f(t + h, q_new) for error estimation
    this->rhs(K_ + Nstages*n, t + h, q_new);
    result[0] = t + h;
}


// StandardRungeKutta implementations
template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::StandardRungeKutta(SOLVER_CONSTRUCTOR(T))
    requires (!is_rich<SP>): Base(ARGS, Nstages+1) {}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::StandardRungeKutta(SOLVER_CONSTRUCTOR(T), EVENTS events)
    requires (is_rich<SP>): Base(ARGS, events, Nstages+1) {}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline std::unique_ptr<Interpolator<T, N>> StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return std::unique_ptr<Interpolator<T, N>>(new StandardLocalInterpolator<T, N>(this->_coef_mat, this->t_old(), d[0], this->old_state_ptr()+2, d+2, this->Nsys(), bdr1, bdr2));
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline void StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, this->_coef_mat.data(), Derived::INTERP_ORDER, this->Nsys());
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::set_coef_matrix_impl() const{
    for (size_t i=0; i<this->Nsys(); i++){
        for (size_t j=0; j<Derived::INTERP_ORDER; j++){
            T sum = 0;
            for (size_t k=0; k<Nstages+1; k++){
                sum += this->_K_true(k, i) * P(k, j);
            }
            this->_coef_mat(i, j) = sum;
        }
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
inline T StandardRungeKutta<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::estimate_error_norm(const T* K, const T* scale, T h) const {
    return _error_norm<T, Nstages>(this->_error_tmp.data(), E.data(), K, h, scale, this->Nsys());
}


// RK45 implementations
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK45<T, N, SP, RhsType, JacType>::RK45(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ARGS){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK45<T, N, SP, RhsType, JacType>::RK45(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ARGS, events){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Atype RK45<T, N, SP, RhsType, JacType>::Amatrix() {
    return {   T(0),        T(0),        T(0),        T(0),        T(0), T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0), T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0), T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0), T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Btype RK45<T, N, SP, RhsType, JacType>::Bmatrix(){
    T q[] = { T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84)};
    typename Base::Btype B(q);
    return B;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Ctype RK45<T, N, SP, RhsType, JacType>::Cmatrix(){

    T q[] = { T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1)};
    typename Base::Ctype C(q);
    return C;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Etype RK45<T, N, SP, RhsType, JacType>::Ematrix() {

    T q[] = { T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40)};
    typename Base::Etype E(q);
    return E;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Ptype RK45<T, N, SP, RhsType, JacType>::Pmatrix() {

    T q[] = {    T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423)};
    typename Base::Ptype P(q, Nstages+1, static_cast<size_t>(4));
    return P;
}


// RK23 implementations
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK23<T, N, SP, RhsType, JacType>::RK23(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ARGS){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK23<T, N, SP, RhsType, JacType>::RK23(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ARGS, events){}
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Atype RK23<T, N, SP, RhsType, JacType>::Amatrix() {
    return { T(0),    T(0),    T(0),
            T(1)/T(2), T(0),    T(0),
            T(0),    T(3)/T(4), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Btype RK23<T, N, SP, RhsType, JacType>::Bmatrix(){
    T q[] = { T(2)/T(9),
            T(1)/T(3),
            T(4)/T(9)};
    typename Base::Btype B(q);
    return B;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Ctype RK23<T, N, SP, RhsType, JacType>::Cmatrix(){

    T q[] = { T(0),
            T(1)/T(2),
            T(3)/T(4)};
    typename Base::Ctype C(q);
    return C;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Etype RK23<T, N, SP, RhsType, JacType>::Ematrix() {
    T q[] = { T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8)};
    typename Base::Etype E(q);
    return E;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Ptype RK23<T, N, SP, RhsType, JacType>::Pmatrix() {
    T q[] = { T(1),   -T(4)/T(3),  T(5)/T(9),
        T(0),    T(1),      -T(2)/T(3),
        T(0),    T(4)/T(3), -T(8)/T(9),
        T(0),   -T(1),       T(1)};
    typename Base::Ptype P(q, Nstages+1, static_cast<size_t>(3));
    return P;
}

} // namespace ode

#endif
