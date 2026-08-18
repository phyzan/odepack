#ifndef DOPRI_IMPL_HPP
#define DOPRI_IMPL_HPP

#include "DOPRI.hpp"
#include "ndspan/ndtools.hpp"

namespace ode{

// ============================================================================
// Shared explicit Runge-Kutta building blocks
// ============================================================================

namespace detail{

template<typename T>
void rk_interp_matrix(T* coef_mat, const T* K, const T* P, size_t Nstages, size_t order, size_t n){
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < order; j++){
            T sum = 0;
            for (size_t k = 0; k <= Nstages; k++){
                sum += K[k*n + i] * P[k*order + j];
            }
            coef_mat[i*order + j] = sum;
        }
    }
}



template<typename T, typename StepFn>
StepResult rk_adapt_step(T* res, const T* state, T* K, size_t n, size_t fsal_row,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn){
    T& habs = res[1];
    habs = state[1];
    T* q_new = res + 2;

    ndspan::copy_array(K, K + fsal_row*n, n); // FSAL: reuse the last stage of the previous step

    bool step_accepted = false;
    T factor;
    while (!step_accepted){
        const T h = habs * direction;
        const T err_norm = step_fn(res, state, h);

        if (err_norm <= 1){
            step_accepted = true;
            if (2*err_norm < 1){
                const auto& err_clamped = max_ref(err_norm, min_err);
                set_min(factor, max_factor, safety * pow(err_clamped, inc_exp));
            } else {
                factor = 1;
            }
        } else {
            set_max(factor, min_factor, safety * pow(err_norm, err_exp));
        }

        if (!all_are_finite(q_new, n)){
            return StepResult::INF_ERROR;
        } else if (habs < min_step_abs){
            return StepResult::TINY_STEP_ERROR;
        } else if (!resize_step(factor, habs, min_step, max_step)){
            break;
        }
    }
    return StepResult::Success;
}

} // namespace ode::detail

// ============================================================================
// RK23
// ============================================================================

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK23<T, N, SP, OdeType, Derived>::RK23(MAIN_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args)),
      K_(Nstages+1, q0.size()), df_tmp_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(K_.data() + Nstages*q0.size(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK23<T, N, SP, OdeType, Derived>::RK23(MAIN_CONSTRUCTOR(T), EventList<T> events) requires (traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args), std::move(events)),
      K_(Nstages+1, q0.size()), df_tmp_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(K_.data() + Nstages*q0.size(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Integrator RK23<T, N, SP, OdeType, Derived>::method() const{
    return Integrator::RK23;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K_.fill(0);
    mat_is_set_ = false;
    const T* state = this->new_state_ptr();
    this->rhs(K_.data() + Nstages*this->Nsys(), state[0], state+2);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    // freeze the interpolation coefficients (from t_old to t) before K is overwritten
    this->set_coef_matrix();
    this->rhs(K_.data() + Nstages*this->Nsys(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set_){
        detail::rk_interp_matrix(coef_mat_.data(), K_.data(), P.data(), Nstages, INTERP_ORDER, this->Nsys());
        mat_is_set_ = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK23<T, N, SP, OdeType, Derived>::step_impl(T* result, const T* state, const T& h){
    const T& t = state[0];
    T* __restrict__ q_new = result + 2;
    T* __restrict__ K = K_.data();
    T* __restrict__ r = df_tmp_.data();
    const T* __restrict__ q = state + 2;
    const size_t n = this->Nsys();

    const T rtol = this->rtol(), atol = this->atol();
    const T habs = abs<T>(h);

    const T* __restrict__ K0 = K;
    T* __restrict__       K1 = K +   n;
    T* __restrict__       K2 = K + 2*n;
    T* __restrict__       K3 = K + 3*n;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(1,0)*K0[j]); }
    this->rhs(K1, t + C[1]*h, r);

    // Stage 3 (a31 = 0, so K0 absent)
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(2,1)*K1[j]); }
    this->rhs(K2, t + C[2]*h, r);

    // Solution update
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(1)*K1[j] + B(2)*K2[j]);
    }

    // FSAL: K3 = f(t+h, q_new)
    this->rhs(K3, t + h, q_new);
    result[0] = t + h;

    // Error norm
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[1]*K1[j] + E[2]*K2[j] + E[3]*K3[j]);
        const auto scale = atol + rtol * (abs<T>(q[j]) + abs<T>(K0[j]) * habs);
        err_max = ndspan::max<T>(err_max, abs<T>(err) / scale);
    }
    return err_max;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK23<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    mat_is_set_ = false;
    return detail::rk_adapt_step(res, state, K_.data(), this->Nsys(), Nstages,
                          this->min_step(), this->max_step(), this->MIN_STEP,
                          this->SAFETY, this->MAX_FACTOR, this->MIN_FACTOR,
                          ERR_EXP, INC_EXP, MIN_ERR, this->direction(),
                          [this](T* r, const T* s, const T& h){ return this->step_impl(r, s, h); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, coef_mat_.data(), INTERP_ORDER, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK23<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return [cm=this->coef_mat_, t1=this->t_old(), t2=d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, this->Nsys()), y2=Array1D<T, N>(d+2, this->Nsys()), n=this->Nsys()](T* out, const T& t){
        coef_mat_interp(out, t, t1, t2, y1.data(), y2.data(), cm.data(), INTERP_ORDER, n);
    };
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Atype RK23<T, N, SP, OdeType, Derived>::Amatrix() {
    return {T(0),   T(0),      T(0),
            T(1)/2, T(0),      T(0),
            T(0),   T(3)/T(4), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Btype RK23<T, N, SP, OdeType, Derived>::Bmatrix(){
    return {T(2)/9,
            T(1)/3,
            T(4)/9};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Ctype RK23<T, N, SP, OdeType, Derived>::Cmatrix(){
    return {T(0),
            T(1)/T(2),
            T(3)/T(4)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Etype RK23<T, N, SP, OdeType, Derived>::Ematrix() {
    return {T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Ptype RK23<T, N, SP, OdeType, Derived>::Pmatrix() {
    return {T(1),  -T(4)/T(3),  T(5)/T(9),
            T(0),   T(1),      -T(2)/T(3),
            T(0),   T(4)/T(3), -T(8)/T(9),
            T(0),  -T(1),       T(1)};
}


// ============================================================================
// RK45
// ============================================================================

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK45<T, N, SP, OdeType, Derived>::RK45(MAIN_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args)),
      K_(Nstages+1, q0.size()), df_tmp_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(K_.data() + Nstages*q0.size(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK45<T, N, SP, OdeType, Derived>::RK45(MAIN_CONSTRUCTOR(T), EventList<T> events) requires (traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(args), std::move(events)),
      K_(Nstages+1, q0.size()), df_tmp_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(K_.data() + Nstages*q0.size(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Integrator RK45<T, N, SP, OdeType, Derived>::method() const{
    return Integrator::RK45;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K_.fill(0);
    mat_is_set_ = false;
    const T* state = this->new_state_ptr();
    this->rhs(K_.data() + Nstages*this->Nsys(), state[0], state+2);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    this->set_coef_matrix();
    this->rhs(K_.data() + Nstages*this->Nsys(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set_){
        detail::rk_interp_matrix(coef_mat_.data(), K_.data(), P.data(), Nstages, INTERP_ORDER, this->Nsys());
        mat_is_set_ = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK45<T, N, SP, OdeType, Derived>::step_impl(T* result, const T* state, const T& h){
    const T& t = state[0];
    T* __restrict__ q_new = result + 2;
    T* __restrict__ K = K_.data();
    T* __restrict__ r = df_tmp_.data();
    const T* __restrict__ q = state + 2;
    const size_t n = this->Nsys();

    const T rtol = this->rtol(), atol = this->atol();
    const T habs = abs<T>(h);

    const T* __restrict__ K0 = K;
    T* __restrict__       K1 = K +   n;
    T* __restrict__       K2 = K + 2*n;
    T* __restrict__       K3 = K + 3*n;
    T* __restrict__       K4 = K + 4*n;
    T* __restrict__       K5 = K + 5*n;
    T* __restrict__       K6 = K + 6*n;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(1,0)*K0[j]); }
    this->rhs(K1, t + C[1]*h, r);

    // Stage 3
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(2,0)*K0[j] + A(2,1)*K1[j]); }
    this->rhs(K2, t + C[2]*h, r);

    // Stage 4
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(3,0)*K0[j] + A(3,1)*K1[j] + A(3,2)*K2[j]); }
    this->rhs(K3, t + C[3]*h, r);

    // Stage 5
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(4,0)*K0[j] + A(4,1)*K1[j] + A(4,2)*K2[j] + A(4,3)*K3[j]); }
    this->rhs(K4, t + C[4]*h, r);

    // Stage 6
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(5,0)*K0[j] + A(5,1)*K1[j] + A(5,2)*K2[j] + A(5,3)*K3[j] + A(5,4)*K4[j]); }
    this->rhs(K5, t + h, r);

    // Solution update (b2 = 0)
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(2)*K2[j] + B(3)*K3[j] + B(4)*K4[j] + B(5)*K5[j]);
    }

    // FSAL: K6 = f(t+h, q_new)
    this->rhs(K6, t + h, q_new);
    result[0] = t + h;

    // Error norm (e2 = 0; scale uses initial derivative K0)
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[2]*K2[j] + E[3]*K3[j] + E[4]*K4[j] + E[5]*K5[j] + E[6]*K6[j]);
        const auto scale = atol + rtol * (abs<T>(q[j]) + abs<T>(K0[j]) * habs);
        err_max = ndspan::max<T>(err_max, abs<T>(err) / scale);
    }
    return err_max;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK45<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    mat_is_set_ = false;
    return detail::rk_adapt_step(res, state, K_.data(), this->Nsys(), Nstages,
                          this->min_step(), this->max_step(), this->MIN_STEP,
                          this->SAFETY, this->MAX_FACTOR, this->MIN_FACTOR,
                          ERR_EXP, INC_EXP, MIN_ERR, this->direction(),
                          [this](T* r, const T* s, const T& h){ return this->step_impl(r, s, h); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, coef_mat_.data(), INTERP_ORDER, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK45<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return [cm=this->coef_mat_, t1=this->t_old(), t2=d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, this->Nsys()), y2=Array1D<T, N>(d+2, this->Nsys()), n=this->Nsys()](T* out, const T& t){
        coef_mat_interp(out, t, t1, t2, y1.data(), y2.data(), cm.data(), INTERP_ORDER, n);
    };
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Atype RK45<T, N, SP, OdeType, Derived>::Amatrix() {
    return {T(0),        T(0),        T(0),        T(0),        T(0), T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0), T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0), T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0), T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Btype RK45<T, N, SP, OdeType, Derived>::Bmatrix(){
    return {T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Ctype RK45<T, N, SP, OdeType, Derived>::Cmatrix(){
    return {T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Etype RK45<T, N, SP, OdeType, Derived>::Ematrix() {
    return {T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Ptype RK45<T, N, SP, OdeType, Derived>::Pmatrix() {
    return {T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423)};
}

} // namespace ode

#endif // DOPRI_IMPL_HPP
