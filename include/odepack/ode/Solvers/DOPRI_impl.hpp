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
void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n){
    // The Nstages+1 stage rows are no longer one contiguous block: row 0 is K0, rows
    // 1..Nstages-1 live in K, and the final (FSAL) row is KF.
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < order; j++){
            T sum = K0[i] * P[j] + KF[i] * P[Nstages*order + j];
            for (size_t k = 1; k < Nstages; k++){
                sum += K[(k-1)*n + i] * P[k*order + j];
            }
            coef_mat[i*order + j] = sum;
        }
    }
}


template<typename T, size_t NSYS, size_t NCOUNT>
class StaticRKScratch{

public:

    StaticRKScratch(size_t nsys){
        assert(nsys == NSYS && "RKScratch: nsys must match template parameter NSYS for fixed-size systems.");
    }

    std::array<T, NSYS*NCOUNT> stage_scratch() const {return std::array<T, NSYS*NCOUNT>{};}
    std::array<T, NSYS> rhs_scratch() const {return std::array<T, NSYS>{};}
    std::array<T, NSYS> fsal_scratch() const {return std::array<T, NSYS>{};}
    std::array<T, NSYS+2> state_scratch() const {return std::array<T, NSYS+2>{};}

};


// Heap-backed stage scratch: runtime-sized systems, and any scalar that is not trivially
// constructible/copyable (mpfr::mpreal and friends), where a fresh stack array per step would
// mean constructing and destroying nsys*NCOUNT heap-owning objects on every attempt.
template<typename T, size_t NSYS, size_t NCOUNT>
class DynamicRKScratch{

public:

    DynamicRKScratch(size_t nsys) : stage_scratch_(nsys * NCOUNT), rhs_scratch_(nsys),
                                    fsal_scratch_(nsys), state_scratch_(nsys + 2) {
        assert(nsys > 0 && "RKScratch: nsys must be greater than zero.");
    }

    std::vector<T>& stage_scratch() const {return stage_scratch_;}
    std::vector<T>& rhs_scratch() const {return rhs_scratch_;}
    std::vector<T>& fsal_scratch() const {return fsal_scratch_;}
    std::vector<T>& state_scratch() const {return state_scratch_;}

private:
    mutable std::vector<T> stage_scratch_;
    mutable std::vector<T> rhs_scratch_;
    mutable std::vector<T> fsal_scratch_;
    mutable std::vector<T> state_scratch_;

};



template<typename T, typename StepFn>
StepResult rk_adapt_step(T* res, const T* state, size_t n,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn){
    T& habs = res[1];
    habs = state[1];
    T* q_new = res + 2;


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

template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
ODEPACK_STEP_ATTR T rk23_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs){
    const size_t n = NSYS ? NSYS : nsys;
    const T& t = state[0];
    T* __restrict__       q_new = result + 2;
    const T* __restrict__ q     = state + 2;

    T* __restrict__ K1 = K;
    T* __restrict__ K2 = K + n;
    T* __restrict__ K3 = KF;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(1,0)*K0[j]); }
    rhs(K1, t + C[1]*h, r);

    // Stage 3 (a31 = 0, so K0 absent)
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(2,1)*K1[j]); }
    rhs(K2, t + C[2]*h, r);

    // Solution update
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(1)*K1[j] + B(2)*K2[j]);
    }

    // FSAL: K3 = f(t+h, q_new)
    rhs(K3, t + h, q_new);
    result[0] = t + h;

    // Error norm
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[1]*K1[j] + E[2]*K2[j] + E[3]*K3[j]);
        const auto scale = atol + rtol * (abs<T>(q[j]) + abs<T>(K0[j] * h));
        err_max = ndspan::max<T>(err_max, abs<T>(err) / scale);
    }
    return err_max;
}

template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
ODEPACK_STEP_ATTR T rk45_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs){
    // NSYS is a constant, so this folds to a compile-time bound for fixed-size systems and
    // keeps the loops unrolled whether or not the call itself gets inlined.
    const size_t n = NSYS ? NSYS : nsys;
    const T& t = state[0];
    T* __restrict__       q_new = result + 2;
    const T* __restrict__ q     = state + 2;

    T* __restrict__ K1 = K;
    T* __restrict__ K2 = K +   n;
    T* __restrict__ K3 = K + 2*n;
    T* __restrict__ K4 = K + 3*n;
    T* __restrict__ K5 = K + 4*n;
    T* __restrict__ K6 = KF;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(1,0)*K0[j]); }
    rhs(K1, t + C[1]*h, r);

    // Stage 3
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(2,0)*K0[j] + A(2,1)*K1[j]); }
    rhs(K2, t + C[2]*h, r);

    // Stage 4
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(3,0)*K0[j] + A(3,1)*K1[j] + A(3,2)*K2[j]); }
    rhs(K3, t + C[3]*h, r);

    // Stage 5
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(4,0)*K0[j] + A(4,1)*K1[j] + A(4,2)*K2[j] + A(4,3)*K3[j]); }
    rhs(K4, t + C[4]*h, r);

    // Stage 6
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(5,0)*K0[j] + A(5,1)*K1[j] + A(5,2)*K2[j] + A(5,3)*K3[j] + A(5,4)*K4[j]); }
    rhs(K5, t + h, r);

    // Solution update (b2 = 0)
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(2)*K2[j] + B(3)*K3[j] + B(4)*K4[j] + B(5)*K5[j]);
    }

    // FSAL: K6 = f(t+h, q_new)
    rhs(K6, t + h, q_new);
    result[0] = t + h;

    // Error norm (e2 = 0; scale uses the derivative at the start of the step)
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[2]*K2[j] + E[3]*K3[j] + E[4]*K4[j] + E[5]*K5[j] + E[6]*K6[j]);
        const auto scale = atol + rtol * (abs<T>(q[j]) + abs<T>(K0[j] * h));
        err_max = ndspan::max<T>(err_max, abs<T>(err) / scale);
    }
    return err_max;
}

} // namespace ode::detail

// ============================================================================
// RK23
// ============================================================================

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK23<T, N, SP, OdeType, Derived>::RK23(MAIN_CONSTRUCTOR(T)) requires (!traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir),
      scratch_space(q0.size()), K0_(q0.size()), KF_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK23<T, N, SP, OdeType, Derived>::RK23(MAIN_CONSTRUCTOR(T), EventList<T> events) requires (traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(events)),
      scratch_space(q0.size()), K0_(q0.size()), KF_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Integrator RK23<T, N, SP, OdeType, Derived>::method() const{
    return Integrator::RK23;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K0_.fill(0);
    KF_.fill(0);
    mat_is_set_ = false;
    const T* state = this->new_state_ptr();
    this->rhs(KF_.data(), state[0], state+2);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    // freeze the interpolation coefficients (from t_old to t) before KF_ is overwritten
    this->set_coef_matrix();
    this->rhs(KF_.data(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set_){
        // See RK45::set_coef_matrix: the stage scratch is transient, so the finished step is
        // replayed over the same interval (K0_ and h_last_ still describe it) into private
        // buffers, leaving KF_ - the next step's starting derivative - untouched.
        const size_t n = this->nsys();
        decltype(auto) r_buf     = scratch_space.rhs_scratch();
        decltype(auto) K_buf     = scratch_space.stage_scratch();
        decltype(auto) kf_buf    = scratch_space.fsal_scratch();
        decltype(auto) state_buf = scratch_space.state_scratch();

        detail::rk23_step_impl<N>(state_buf.data(), this->old_state_ptr(), h_last_, n,
                                  K0_.data(), K_buf.data(), kf_buf.data(), r_buf.data(),
                                  this->rtol(), this->atol(), A, B, C, E,
                                  [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });

        detail::rk_interp_matrix(coef_mat_.data(), K_buf.data(), K0_.data(), kf_buf.data(),
                                 P.data(), Nstages, INTERP_ORDER, n);
        mat_is_set_ = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK23<T, N, SP, OdeType, Derived>::step_impl(T* result, const T* state, const T& h){
    decltype(auto) rhs_scratch   = scratch_space.rhs_scratch();
    decltype(auto) stage_scratch = scratch_space.stage_scratch();
    h_last_ = h; // remembered so set_coef_matrix can replay this sweep exactly
    return detail::rk23_step_impl<N>(result, state, h, this->nsys(),
                                     K0_.data(), stage_scratch.data(), KF_.data(), rhs_scratch.data(),
                                     this->rtol(), this->atol(), A, B, C, E,
                                     [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK23<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    mat_is_set_ = false;
    // FSAL: the previous step's last stage becomes this step's first
    copy_array(K0_.data(), KF_.data(), this->nsys());
    return detail::rk_adapt_step(res, state, this->nsys(),
                          this->min_step(), this->max_step(), this->MIN_STEP,
                          this->SAFETY, this->MAX_FACTOR, this->MIN_FACTOR,
                          ERR_EXP, INC_EXP, MIN_ERR, this->direction(),
                          [this](T* r, const T* s, const T& h){ return this->step_impl(r, s, h); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, coef_mat_.data(), INTERP_ORDER, this->nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK23<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return [cm=this->coef_mat_, t1=this->t_old(), t2=d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, this->nsys()), y2=Array1D<T, N>(d+2, this->nsys()), n=this->nsys()](T* out, const T& t){
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
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir),
    scratch_space(q0.size()),
    K0_(q0.size()),
    KF_(q0.size()), coef_mat(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK45<T, N, SP, OdeType, Derived>::RK45(MAIN_CONSTRUCTOR(T), EventList<T> events) requires (traits::is_rich<SP>)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(events)),
      scratch_space(q0.size()),
      K0_(q0.size()),
      KF_(q0.size()), coef_mat(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Integrator RK45<T, N, SP, OdeType, Derived>::method() const{
    return Integrator::RK45;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K0_.fill(0);
    KF_.fill(0);
    mat_is_set = false;
    const T* state = this->new_state_ptr();
    this->rhs(KF_.data(), state[0], state+2);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    this->set_coef_matrix();
    this->rhs(KF_.data(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set){
        // The stages of the step just taken are needed to build the dense-output polynomial,
        // but the step's stage scratch is transient, so replay the sweep over the same
        // interval. K0_ still holds the derivative at the start of that step and h_last_ the
        // step size it used, so the replay reproduces the stages exactly. It writes into its
        // own result/FSAL scratch, leaving KF_ - the next step's starting derivative - alone.
        // Cost: Nstages RHS evaluations, paid only when an interpolation is actually asked for.
        const size_t n = this->nsys();
        decltype(auto) r_buf     = scratch_space.rhs_scratch();
        decltype(auto) K_buf     = scratch_space.stage_scratch();
        decltype(auto) kf_buf    = scratch_space.fsal_scratch();
        decltype(auto) state_buf = scratch_space.state_scratch();

        detail::rk45_step_impl<N>(state_buf.data(), this->old_state_ptr(), h_last_, n,
                               K0_.data(), K_buf.data(), kf_buf.data(), r_buf.data(),
                               this->rtol(), this->atol(), A, B, C, E,
                               [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });

        detail::rk_interp_matrix(coef_mat.data(), K_buf.data(), K0_.data(), kf_buf.data(),
                                 P.data(), Nstages, INTERP_ORDER, n);
        mat_is_set = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK45<T, N, SP, OdeType, Derived>::step_impl(T* result, const T* state, const T& h){
    decltype(auto) rhs_scratch   = scratch_space.rhs_scratch();
    decltype(auto) stage_scratch = scratch_space.stage_scratch();
    h_last_ = h; // remembered so set_coef_matrix can replay this sweep exactly
    return detail::rk45_step_impl<N>(result, state, h, this->nsys(),
                                  K0_.data(), stage_scratch.data(), KF_.data(), rhs_scratch.data(),
                                  this->rtol(), this->atol(), A, B, C, E,
                                  [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK45<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    mat_is_set = false;
    copy_array(K0_.data(), KF_.data(), this->nsys());
    return detail::rk_adapt_step(res, state, this->nsys(),
                          this->min_step(), this->max_step(), this->MIN_STEP,
                          this->SAFETY, this->MAX_FACTOR, this->MIN_FACTOR,
                          ERR_EXP, INC_EXP, MIN_ERR, this->direction(),
                          [this](T* r, const T* s, const T& h){ return this->step_impl(r, s, h); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, coef_mat.data(), INTERP_ORDER, this->nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK45<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return [cm=this->coef_mat, t1=this->t_old(), t2=d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, this->nsys()), y2=Array1D<T, N>(d+2, this->nsys()), n=this->nsys()](T* out, const T& t){
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
