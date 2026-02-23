#ifndef DOPRI_IMPL_HPP
#define DOPRI_IMPL_HPP

#include "DOPRI.hpp"

namespace ode{

// RungeKuttaMainBase implementations
template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::RungeKuttaMainBase(SOLVER_CONSTRUCTOR(T))
    requires (!is_rich<SP>): Base(ARGS), K_(K_ROWS, nsys), _df_tmp(nsys), _coef_mat(nsys, Derived::INTERP_ORDER) {
    // Compute K[Nstages] = f(t0, q0) for FSAL
    if (q0 != nullptr){
        this->rhs(K_.data()+Nstages*nsys, t0, q0);
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::RungeKuttaMainBase(SOLVER_CONSTRUCTOR(T), EVENTS events)
    requires (is_rich<SP>): Base(ARGS, events), K_(K_ROWS, nsys), _df_tmp(nsys), _coef_mat(nsys, Derived::INTERP_ORDER) {
    // Compute K[Nstages] = f(t0, q0) for FSAL
    if (q0 != nullptr){
        this->rhs(K_.data()+Nstages*nsys, t0, q0);
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    K_.set(0);
    _mat_is_set = false;
    // Compute K[Nstages] = f(t0, q0) for FSAL
    const T* state = this->new_state_ptr();
    this->rhs(K_.data()+Nstages*this->Nsys(), state[0], state+2);
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
    // Re-compute K[Nstages] = f(t, q) after restarting at intermediate time (e.g. because of discontinuity due to masked event)
    Base::re_adjust_impl(new_vector);

    //prepate the coef matrix for interpolation
    //before altering K, so that interpolation is valid
    //from t_old to t
    this->set_coef_matrix();
    //copy the new state vector into the last stage of K
    this->rhs(K_.data() + Nstages*this->Nsys(), this->t(), new_vector);
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
void RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::set_coef_matrix_impl() const{
    THIS->set_coef_matrix_impl();
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
 void RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::set_coef_matrix() const{
    if (!this->_mat_is_set){
        this->set_coef_matrix_impl();
        this->_mat_is_set = true;
    }
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
T RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::step_impl(T* result, const T* state, const T& h){
    const T& t = state[0];
    T* __restrict__ q_new = result + 2;
    T* __restrict__ K = this->K_.data();
    T* __restrict__ r = this->_df_tmp.data();
    const T* __restrict__ q = state + 2;
    const auto& B_ = THIS->B;
    const auto& C_ = THIS->C;
    const auto& A_ = THIS->A;

    for_each<0, Nstages-1>([&]<size_t I>() LAMBDA_INLINE{
        for (size_t j = 0; j < this->Nsys(); j++) {
            r[j] = q[j] + h * EXPAND_SUM(T, I+1, J,
                A_(I+1, J)*K[J*this->Nsys()+j]
            );
        }
        this->rhs(K + (I+1)*this->Nsys(), t + C_[I+1]*h, r);
    });

    for (size_t j=0; j<this->Nsys(); j++){
        q_new[j] = q[j] + h * EXPAND_SUM(T, Nstages, I,
            B_(I)*K[I*this->Nsys()+j]
        );
    }

    // Final: K[Nstages] = f(t + h, q_new) for error estimation and FSAL
    this->rhs(K + Nstages*this->Nsys(), t + h, q_new);
    result[0] = t + h;
    return THIS->estimate_error_norm(K, q, q_new, this->rtol(), this->atol(), h);
}


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, size_t K_ROWS, SolverPolicy SP, typename RhsType, typename JacType>
StepResult RungeKuttaMainBase<Derived, T, N, Nstages, Norder, K_ROWS, SP, RhsType, JacType>::adapt_impl(T* res, const T* state){
    this->_mat_is_set = false;
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const size_t n = this->Nsys();

    T& habs = res[1];
    habs = state[1];
    T h, err_norm, factor, _factor;
    T* q_new = res+2;

    bool step_accepted = false;
    T* K = K_.data();
    copy_array(K, K + Nstages*this->Nsys(), this->Nsys()); //FSAL: K[0] for next step = K[Nstages] from this step

    while (!step_accepted){
        h = habs * this->direction();
        err_norm = THIS->step_impl(res, state, h);

        if (err_norm <= 1){
            // Accept step
            step_accepted = true;
            if (2*err_norm < 1) {
                T err_clamped = max(err_norm, MIN_ERR);
                _factor = this->SAFETY * pow(err_clamped, INC_EXP);
                factor = min(this->MAX_FACTOR, _factor);
            } else {
                factor = 1;
            }
        } else {
            // Reject step
            _factor = this->SAFETY * pow(err_norm, ERR_EXP);
            factor = max(this->MIN_FACTOR, _factor);
        }

        if (!all_are_finite(q_new, n)){
            return StepResult::INF_ERROR;
        }else if (habs < this->MIN_STEP){
            return StepResult::TINY_STEP_ERROR;
        }else if (!resize_step(factor, habs, h_min, max_step)){
            break;
        }
    }
    return StepResult::Success;
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
T StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::estimate_error_norm(const T* K, const T* q, const T* q_new, const T& rtol, const T& atol, const T& h) const {
    // Boost-style error calculation: max norm with |x| + |dxdt|*|h| scaling
    const T habs = abs(h);
    T err_max = 0;
    for (size_t j = 0; j < this->Nsys(); j++) {

        T err_total = h * EXPAND_SUM(T, Nstages+1, I,
            THIS->E[I] * this->K_[I*this->Nsys()+j]
        );
        T scale = atol + rtol * (abs(q[j]) + abs(this->K_[j]) * habs);
        err_max = max(err_max, abs(err_total) / scale);
    }
    return err_max;
}


// StandardRungeKutta implementations
template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::StandardRungeKuttaBase(SOLVER_CONSTRUCTOR(T))
    requires (!is_rich<SP>): Base(ARGS) {}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
    StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::StandardRungeKuttaBase(SOLVER_CONSTRUCTOR(T), EVENTS events)
    requires (is_rich<SP>): Base(ARGS, events) {}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
 std::unique_ptr<Interpolator<T, N>> StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return std::unique_ptr<Interpolator<T, N>>(new StandardLocalInterpolator<T, N>(this->_coef_mat, this->t_old(), d[0], this->old_state_ptr()+2, d+2, this->Nsys(), bdr1, bdr2));
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
 void StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, this->_coef_mat.data(), Derived::INTERP_ORDER, this->Nsys());
}

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder, SolverPolicy SP, typename RhsType, typename JacType>
void StandardRungeKuttaBase<Derived, T, N, Nstages, Norder, SP, RhsType, JacType>::set_coef_matrix_impl() const{
    const auto& P = THIS->P;
    for (size_t i=0; i<this->Nsys(); i++){
        for (size_t j=0; j<Derived::INTERP_ORDER; j++){
            T sum = 0;
            for (size_t k=0; k<Nstages+1; k++){
                sum += this->K_(k, i) * P(k, j);
            }
            this->_coef_mat(i, j) = sum;
        }
    }
}


// RK45 implementations
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK45<T, N, SP, RhsType, JacType>::RK45(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ARGS){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK45<T, N, SP, RhsType, JacType>::RK45(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ARGS, events){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Atype RK45<T, N, SP, RhsType, JacType>::Amatrix() {
    return {T(0),        T(0),        T(0),        T(0),        T(0), T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0), T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0), T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0), T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Btype RK45<T, N, SP, RhsType, JacType>::Bmatrix(){
    return {T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Ctype RK45<T, N, SP, RhsType, JacType>::Cmatrix(){
    return {T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK45<T, N, SP, RhsType, JacType>::Base::Etype RK45<T, N, SP, RhsType, JacType>::Ematrix() {
    return {T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK45<T, N, SP, RhsType, JacType>::Ptype RK45<T, N, SP, RhsType, JacType>::Pmatrix() {
    return {T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423)};
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
T RK45<T, N, SP, RhsType, JacType>::step_impl(T* result, const T* state, const T& h) {

    // Hardcoded for performance, no algorithmic difference from Base.
    const T& t = state[0];
    T* __restrict__ q_new = result + 2;
    T* __restrict__ K = this->K_.data();
    T* __restrict__ r = this->_df_tmp.data();
    const T* __restrict__ q = state + 2;
    const size_t n = this->Nsys();

    // A coefficients (lower-triangular, 1-based stage notation)
    const T a21 = this->A(1,0);
    const T a31 = this->A(2,0), a32 = this->A(2,1);
    const T a41 = this->A(3,0), a42 = this->A(3,1), a43 = this->A(3,2);
    const T a51 = this->A(4,0), a52 = this->A(4,1), a53 = this->A(4,2), a54 = this->A(4,3);
    const T a61 = this->A(5,0), a62 = this->A(5,1), a63 = this->A(5,2), a64 = this->A(5,3), a65 = this->A(5,4);
    // B weights (b2 = 0)
    const T b1 = this->B(0), b3 = this->B(2), b4 = this->B(3), b5 = this->B(4), b6 = this->B(5);
    // C nodes
    const T c2 = this->C[1], c3 = this->C[2], c4 = this->C[3], c5 = this->C[4];
    // E error coefficients (e2 = 0)
    const T e1 = this->E[0], e3 = this->E[2], e4 = this->E[3], e5 = this->E[4], e6 = this->E[5], e7 = this->E[6];

    const T rtol = this->rtol(), atol = this->atol();
    const T habs = abs(h);

    const T* __restrict__ K0 = K;
    T* __restrict__       K1 = K +   n;
    T* __restrict__       K2 = K + 2*n;
    T* __restrict__       K3 = K + 3*n;
    T* __restrict__       K4 = K + 4*n;
    T* __restrict__       K5 = K + 5*n;
    T* __restrict__       K6 = K + 6*n;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a21*K0[j]); }
    this->rhs(K1, t + c2*h, r);

    // Stage 3
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a31*K0[j] + a32*K1[j]); }
    this->rhs(K2, t + c3*h, r);

    // Stage 4
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a41*K0[j] + a42*K1[j] + a43*K2[j]); }
    this->rhs(K3, t + c4*h, r);

    // Stage 5
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a51*K0[j] + a52*K1[j] + a53*K2[j] + a54*K3[j]); }
    this->rhs(K4, t + c5*h, r);

    // Stage 6
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a61*K0[j] + a62*K1[j] + a63*K2[j] + a64*K3[j] + a65*K4[j]); }
    this->rhs(K5, t + h, r);

    // Solution update (b2 = 0)
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (b1*K0[j] + b3*K2[j] + b4*K3[j] + b5*K4[j] + b6*K5[j]);
    }

    // FSAL: K6 = f(t+h, q_new)
    this->rhs(K6, t + h, q_new);
    result[0] = t + h;

    // Error norm (e2 = 0; scale uses initial derivative K0)
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const T err   = h * (e1*K0[j] + e3*K2[j] + e4*K3[j] + e5*K4[j] + e6*K5[j] + e7*K6[j]);
        const T scale = atol + rtol * (abs(q[j]) + abs(K0[j]) * habs);
        err_max = max(err_max, abs(err) / scale);
    }
    return err_max;
}


// RK23 implementations
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK23<T, N, SP, RhsType, JacType>::RK23(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ARGS){}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RK23<T, N, SP, RhsType, JacType>::RK23(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ARGS, events){}
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Atype RK23<T, N, SP, RhsType, JacType>::Amatrix() {
    return {T(0),       T(0),       T(0),
            T(1)/2,     T(0),       T(0),
            T(0),       T(3)/T(4),  T(0)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Btype RK23<T, N, SP, RhsType, JacType>::Bmatrix(){
    return {T(2)/9,
            T(1)/3,
            T(4)/9};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Ctype RK23<T, N, SP, RhsType, JacType>::Cmatrix(){
    return { T(0),
            T(1)/T(2),
            T(3)/T(4)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK23<T, N, SP, RhsType, JacType>::Base::Etype RK23<T, N, SP, RhsType, JacType>::Ematrix() {
    return {T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
constexpr typename RK23<T, N, SP, RhsType, JacType>::Ptype RK23<T, N, SP, RhsType, JacType>::Pmatrix() {
    return {T(1),   -T(4)/T(3),  T(5)/T(9),
            T(0),    T(1),      -T(2)/T(3),
            T(0),    T(4)/T(3), -T(8)/T(9),
            T(0),   -T(1),       T(1)};
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
T RK23<T, N, SP, RhsType, JacType>::step_impl(T* result, const T* state, const T& h) {
    const T& t = state[0];
    // Hardcoded for performance, no algorithmic difference from Base.
    T* __restrict__ q_new = result + 2;
    T* __restrict__ K = this->K_.data();
    T* __restrict__ r = this->_df_tmp.data();
    const T* __restrict__ q = state + 2;
    const size_t n = this->Nsys();

    // A coefficients: a21=1/2, a31=0 (so K0 absent in stage 3), a32=3/4
    const T a21 = this->A(1,0);
    const T a32 = this->A(2,1);
    // B weights: all nonzero
    const T b1 = this->B(0), b2 = this->B(1), b3 = this->B(2);
    // C nodes
    const T c2 = this->C[1], c3 = this->C[2];
    // E error coefficients: all nonzero
    const T e1 = this->E[0], e2 = this->E[1], e3 = this->E[2], e4 = this->E[3];

    const T rtol = this->rtol(), atol = this->atol();
    const T habs = abs(h);

    const T* __restrict__ K0 = K;
    T* __restrict__       K1 = K +   n;
    T* __restrict__       K2 = K + 2*n;
    T* __restrict__       K3 = K + 3*n;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a21*K0[j]); }
    this->rhs(K1, t + c2*h, r);

    // Stage 3 (a31 = 0, so K0 absent)
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (a32*K1[j]); }
    this->rhs(K2, t + c3*h, r);

    // Solution update
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (b1*K0[j] + b2*K1[j] + b3*K2[j]);
    }

    // FSAL: K3 = f(t+h, q_new)
    this->rhs(K3, t + h, q_new);
    result[0] = t + h;

    // Error norm
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const T err   = h * (e1*K0[j] + e2*K1[j] + e3*K2[j] + e4*K3[j]);
        const T scale = atol + rtol * (abs(q[j]) + abs(K0[j]) * habs);
        err_max = std::max(err_max, abs(err) / scale);
    }
    return err_max;
}

} // namespace ode

#endif // DOPRI_IMPL_HPP