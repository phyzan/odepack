#ifndef BDF_HPP
#define BDF_HPP

#include "../Core/RichBase.hpp"

namespace ode{

constexpr size_t MAX_ORDER = 5;

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
    Array1D<T, MAX_ORDER+1> KAPPA;
    Array1D<T, MAX_ORDER+1> GAMMA;
    Array1D<T, MAX_ORDER+1> ALPHA;
    Array1D<T, MAX_ORDER+1> ERR_CONST;

    BDFCONSTS();
};

struct NewtConv{
    bool converged;
    size_t n_iter;
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

    using Base = BaseDispatcher<BDF<T, N, SP, RhsType, JacType>, T, N, SP, RhsType, JacType>;
    using Dlike = Array2D<T, 0, N>;
    struct None{};

public:

    static constexpr size_t NEWTON_MAXITER = 4;
    // =============== OVERRIDEN =================
    static constexpr const char* name = "BDF";
    static constexpr bool IS_IMPLICIT = true;
    static constexpr int ERR_EST_ORDER = 1;

    inline std::unique_ptr<Interpolator<T, N>>  state_interpolator(int bdr1, int bdr2) const;
    void                                        adapt_impl(T* res);
    inline void                                 interp_impl(T* result, const T& t) const;
    inline void                                 reset_impl();
    void                                        re_adjust_impl(const T* new_vector);
    bool                                        validate_ics_impl(T t0, const T* q0) const;
    // ===========================================

    DEFAULT_RULE_OF_FOUR(BDF)

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>) : BDF(ARGS, None()) {}

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>) : BDF(ARGS, None(), events) {}

private:

    template<typename... Type>
    BDF(MAIN_CONSTRUCTOR(T), None, Type&&... extras);

    inline void                                 _reset_impl_alone();

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


// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

template<typename T, size_t N>
LUResult<T, N>::LUResult(size_t Nsys) : LU(Nsys, Nsys), piv(Nsys) {}

template<typename T, size_t N>
void LUResult<T, N>::lu_factor(const JacMat<T, N>& A_input){
    size_t n = A_input.Nrows();
    JacMat<T, N>& A = LU;

    // Copy input to output
    copy_array(A.data(), A_input.data(), A_input.size());

    // Initialize pivot array as identity
    for (size_t i = 0; i < n; ++i) {
        piv[i] = i;
    }

    for (size_t k = 0; k < n - 1; ++k) {
        // Find pivot row
        size_t p = k;
        T max_val = abs(A(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (abs(A(i, k)) > max_val) {
                max_val = abs(A(i, k));
                p = i;
            }
        }

        // Check singular
        if (max_val < 1e-15) {
            throw std::runtime_error("Matrix is singular or nearly singular");
        }

        // Record and apply pivot
        if (p != k) {
            std::swap(piv[k], piv[p]);  // Record which row k was swapped with
            for (size_t c = 0; c < n; c++) {
                std::swap(A(k, c), A(p, c));
            }
        }

        // Eliminate below pivot
        for (size_t i = k + 1; i < n; ++i) {
            A(i, k) /= A(k, k);
            for (size_t j = k + 1; j < n; ++j) {
                A(i, j) -= A(i, k) * A(k, j);
            }
        }
    }
}

template<typename T, size_t N>
void LUResult<T, N>::lu_solve(T* x, const T* b) const{
    size_t n = piv.size();

    // Apply permutation: x = P*b
    for (size_t i = 0; i < n; ++i) {
        x[i] = b[piv[i]];
    }

    // Forward substitution: L * y = P*b
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            x[i] -= LU(i, j) * x[j];
        }
    }

    // Backward substitution: U * x = y
    for (size_t i = n; i-- > 0; ) {
        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= LU(i, j) * x[j];
        }
        x[i] /= LU(i, i);
    }
}

template<typename T>
Array1D<T> arange(size_t a, size_t b){
    Array1D<T> res(b-a);
    #pragma omp simd
    for (size_t i=0; i<(b-a); i++){
        res[i] = a+i;
    }
    return res;
}

template<typename T>
void cumprod(T* res, const T* x, size_t size){
    res[0] = x[0];
    for (size_t i=1; i<size; i++){
        res[i] = res[i-1]*x[i];
    }
}

template<typename T>
BDFCONSTS<T>::BDFCONSTS(){
    KAPPA = {0, -T(185)/1000, -T(1)/9, -T(823)/10000, -T(415)/10000, 0};
    GAMMA[0] = 0;
    T cumulative = 0;
    for (size_t i = 1; i < MAX_ORDER+1; ++i) {
        cumulative += T(1) / i;
        GAMMA[i] = cumulative;
    }

    for (size_t i=0; i<MAX_ORDER+1; i++){
        ALPHA[i] = (1-KAPPA[i])*GAMMA[i];
        ERR_CONST[i] = KAPPA[i]*GAMMA[i] + T(1)/(1+i);
    }
}

template<typename T>
void compute_R(T* R, size_t order, T factor) {
    size_t n = order+1;
    // Initialize first row to 1
    for (size_t j=0; j<n; j++){
        R[j] = 1;
    }

    // Compute M[i,j] = (i-1 - factor*j) / i for i,j >= 1, then cumulative product
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            R[i*n+j] = R[(i-1)*n + j]*(i-1 - factor * j) / i;
        }
    }
}

template<typename T>
void bdf_interp(T* result, const T& t, const T& t2, const T& h, const T* D, size_t order, size_t size){
    for (size_t i=0; i<size; i++){
        T p = 1;
        T sum = 0;
        for (size_t j=0; j<order; j++){
            p *= (t - (t2 - h*j))/(h*(1+j));
            sum += D[(j+1)*size+i] * p;
        }
        result[i] = D[i] + sum;
    }
}

template<typename T, size_t N>
BDFInterpolator<T, N>::BDFInterpolator(const T& t, const T* q, size_t nsys) : LocalInterpolator<T, N>(t, q, nsys) {}

template<typename T, size_t N>
BDFInterpolator<T, N>::BDFInterpolator(const Array2D<T, 0, N>& D, size_t order, const T* state1, const T* state2, size_t nsys, int bdr1, int bdr2) : LocalInterpolator<T, N>(state1[0], state2[0], state1+2, state2+2, nsys, bdr1, bdr2), _order(order), _D(order+1, D.Ncols()) {
    int dir = sgn(state1[0], state2[0]);
    _t2 = state2[0];
    _h = state2[1]*dir;
    copy_array(_D.data(), D.data(), _D.size()); //keeping the first (order+1) rows
}

template<typename T, size_t N>
size_t BDFInterpolator<T, N>::order() const {
    return _order;
}

template<typename T, size_t N>
BDFInterpolator<T, N>* BDFInterpolator<T, N>::clone() const {
    return new BDFInterpolator<T, N>(*this);
}

template<typename T, size_t N>
void BDFInterpolator<T, N>::_call_impl(T* result, const T& t) const {
    bdf_interp(result, t, _t2, _h, _D.data(), _order, this->array_size());
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
template<typename... Type>
BDF<T, N, SP, RhsType, JacType>::BDF(MAIN_CONSTRUCTOR(T), None, Type&&... extras) : Base(ARGS, extras...), _J(nsys, nsys), _B(nsys, nsys), _LU(nsys), _R((MAX_ORDER+1)*(MAX_ORDER+1)), _U((MAX_ORDER+1)*(MAX_ORDER+1)), _RU((MAX_ORDER+1)*(MAX_ORDER+1)), _f(nsys), _dy(nsys), _b(nsys), _scale(nsys), _ypred(nsys), _psi(nsys), _d(nsys), _error(nsys), _error_m(nsys), _error_p(nsys) {
    
    if (rtol == 0){
        rtol = 100*std::numeric_limits<T>::epsilon();
#ifndef NO_ODE_WARN
        std::cerr << "Warning: rtol=0 not allowed in the BDF method. Setting rtol = " << rtol << std::endl;
#endif
    }
    _newton_tol = std::max(10 * std::numeric_limits<T>::epsilon() / rtol, std::min(T(3)/100, pow(rtol, T(1)/T(2))));

    if (!this->is_dead() && q0 != nullptr){
        if (this->validate_ics_impl(t0, q0)){
            this->_reset_impl_alone();
        }else{
            this->kill("Initial Jacobian contains nan or inf");
        }
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector) {
    Base::re_adjust_impl(new_vector);
    copy_array(_D[2].data(), _D[_idx_D].data(), this->Nsys());
    copy_array(_D[0].data(), new_vector, this->Nsys());
    this->rhs(_D[0].data()+this->Nsys(), this->t(), new_vector);
    for (size_t i=0; i<this->Nsys(); i++){
        _D[0][i+this->Nsys()] *= this->stepsize() * this->direction();
    }
    this->jac(_J.data(), this->t(), new_vector);
    _order = 1;
    _n_eq_steps = 0;
    _valid_LU = false;
    _idx_D = 0;
    interp_idx = 2;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BDF<T, N, SP, RhsType, JacType>::validate_ics_impl(T t0, const T* q0) const{

    // use _B as it is a dummy variable
    if (Base::validate_ics_impl(t0, q0)){
        this->jac(_B.data(), t0, q0);
        return all_are_finite(_B.data(), _B.size());
    }else {
        return false;
    }
}


template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    this->_reset_impl_alone();
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::_reset_impl_alone(){
    T t0 = this->ics().t();
    T h0 = this->ics().habs() * this->direction();
    const T* q0 = this->ics().vector();
    Array1D<T, N> f(this->Nsys());
    this->rhs(f.data(), t0, q0);

    for (size_t k = 0; k<_D.size(); k++){
        _D[k] = Dlike(MAX_ORDER+3, this->Nsys());
        for (size_t j=0; j<this->Nsys(); j++){
            _D[k](0, j) = q0[j];
            _D[k](1, j) = f[j] * h0;
        }
    }
    this->jac(_J.data(), t0, q0);
    _order = 1;
    _n_eq_steps = 0;
    _valid_LU = false;
    _idx_D = 0;
    interp_idx = 0;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::adapt_impl(T* res){
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const T& atol = this->atol();
    const T& rtol = this->rtol();
    const T* state = this->new_state_ptr();

    const T& t = state[0];
    const T& stepsize = state[1];
    size_t nsys = this->Nsys();

    T& t_new = res[0] = state[0];
    T& habs = res[1];
    T* y_new = res+2;

    copy_array(y_new, state+2, nsys);

    T safety, max_factor, factor, c;
    int delta_order;
    bool converged;
    bool currentjac = false;
    bool step_accepted = false;
    NewtConv conv_result;

    if (stepsize > max_step){
        habs = max_step;
        _change_D(max_step/stepsize);
    }
    else if (stepsize < h_min){
        habs = h_min;
        _change_D(h_min/stepsize);
    }
    else{
        habs = stepsize;
    }


    while(!step_accepted){
        if (habs < h_min){
            this->kill("The stepsize for the BDF method dropped below min_step");
            return;
        }
        else if (habs > max_step){
            _change_D(max_step/habs);
            habs = max_step;
            interp_idx = _idx_D;
            return;
        }

        t_new = t + habs * this->direction();

        _set_prediction(_ypred.data());
        #pragma omp simd
        for (size_t i=0; i<nsys; i++){
            _scale[i] = atol + rtol * abs(_ypred[i]);
        }
        _set_psi(_psi.data());

        converged = false;
        c = habs * this->direction() / BDF_COEFS.ALPHA[_order];
        while (!converged){
            if (!_valid_LU){
                for (size_t i=0; i<nsys; i++){
                    for (size_t j=0; j<nsys; j++){
                        if (i == j){
                            _B(i, j) = 1 - c*_J(i, j);
                        }else{
                            _B(i, j) = -c*_J(i, j);
                        }
                    }
                }
                _LU.lu_factor(_B);
                _valid_LU = true;
            }

            conv_result = _solve_bdf_system(y_new, _ypred.data(), _d, t_new, c, _psi, _LU, _scale);
            converged = conv_result.converged;

            if (!converged){
                if (currentjac){
                    break;
                }
                this->jac(_J.data(), t_new, _ypred.data());
                _valid_LU = false;
                currentjac = true;
            }
        }

        if (!converged){
            factor = T(1)/2;
            habs *= factor;
            _change_D(factor);
            _valid_LU = false;
            continue;
        }

        safety = T(9)/10 * T(2 * NEWTON_MAXITER + 1)/(2 * NEWTON_MAXITER + conv_result.n_iter);
        #pragma omp simd
        for (size_t i=0; i<nsys; i++){
            _scale[i] = atol + rtol * abs(y_new[i]);
            _error[i] = BDF_COEFS.ERR_CONST[_order] * _d[i];
        }
        _error_norms[1] = rms_norm(_error.data(), _scale.data(), nsys);
        if (_error_norms[1] > 1){
            factor = std::max(this->MIN_FACTOR, safety * pow(_error_norms[1], T(-1)/(_order+1)));
            habs *= factor;
            _change_D(factor);
        }
        else{
            step_accepted = true;
        }
    }

    _n_eq_steps++;

    T* D = _D[_idx_D].data();
    for (size_t i=0; i<nsys; i++){
        D[(_order+2)*nsys + i] = _d[i] - D[(_order+1)*nsys + i];
        D[(_order+1)*nsys + i] = _d[i];
    }

    for (size_t i = _order + 1; i -- > 0;) {
        #pragma omp simd
        for (size_t j=0; j<nsys; j++){
            D[i*nsys + j] += D[(i+1)*nsys + j];
        }
    }

    if (_n_eq_steps < _order + 1){
        interp_idx = _idx_D;
        return;
    }

    if (_order > 1){
        for (size_t i=0; i<nsys; i++){
            _error_m[i] = BDF_COEFS.ERR_CONST[_order-1] * D[_order*nsys + i];
        }
        _error_norms[0] = rms_norm(_error_m.data(), _scale.data(), nsys);
    }
    else{
        _error_norms[0] = inf<T>();
    }


    if (_order < MAX_ORDER){
        for (size_t i=0; i<nsys; i++){
            _error_p[i] = BDF_COEFS.ERR_CONST[_order+1] * D[(_order+2)*nsys + i];
        }
        _error_norms[2] = rms_norm(_error_p.data(), _scale.data(), nsys);
    }
    else{
        _error_norms[2] = inf<T>();
    }

    delta_order = -1;
    max_factor = pow(_error_norms[0], T(-1)/(_order));
    for (int i=1; i<3; i++){
        T tmp = pow(_error_norms[i], T(-1)/(_order+i));
        if (tmp > max_factor){
            max_factor = tmp;
            delta_order = i - 1;
        }
    }

    _order += delta_order;

    factor = std::min(this->MAX_FACTOR, safety * max_factor);
    habs *= factor;
    _change_D(factor);
    _valid_LU = false;
    interp_idx = _idx_D;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline std::unique_ptr<Interpolator<T, N>> BDF<T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    return std::unique_ptr<Interpolator<T, N>>(new BDFInterpolator<T, N>(_D[interp_idx], _order, this->old_state_ptr(), this->interp_new_state_ptr(), this->Nsys(), bdr1, bdr2));
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void BDF<T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    bdf_interp(result, t, this->interp_new_state_ptr()[0], this->stepsize()*this->direction(), _D[interp_idx].data(), _order, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
NewtConv BDF<T, N, SP, RhsType, JacType>::_solve_bdf_system(T* y, const T* y_pred, Array1D<T, N>& d, const T& t_new, const T& c, const Array1D<T, N>& psi, const LUResult<T, N>& LU, const Array1D<T, N>& scale){
    d.set(0);
    size_t n = this->Nsys();
    copy_array(y, y_pred, n);
    T dy_norm = 0;
    T dy_norm_old = 0;
    T rate = 0;
    bool converged = false;
    size_t j=0;
    for (size_t k=0; k<NEWTON_MAXITER; k++){
        this->rhs(_f.data(), t_new, y);
        if (!all_are_finite(_f.data(), n)){
            break;
        }

        #pragma omp simd
        for (size_t i=0; i<n; i++){
            _b[i] = c * _f[i] - psi[i] - d[i];
        }
        LU.lu_solve(_dy.data(), _b.data());
        dy_norm = rms_norm(_dy.data(), scale.data(), n);

        if (dy_norm_old == 0){
            rate = 0;
        }
        else{
            rate = dy_norm/dy_norm_old;
        }

        if (rate != 0 && (rate >= 1 || (pow(rate, NEWTON_MAXITER-k)*dy_norm > _newton_tol * (1-rate)))){
            break;
        }

        #pragma omp simd
        for (size_t i=0; i<n; i++){
            y[i] += _dy[i];
            d[i] += _dy[i];
        }

        if (dy_norm == 0 || ((rate != 0) && (rate * dy_norm < _newton_tol * (1-rate)))){
            converged = true;
            break;
        }

        dy_norm_old = dy_norm;
        j++;
    }

    return {.converged = converged, .n_iter = j+1};

}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::_change_D(const T& factor){
    T* R = _R.data();
    T* U = _U.data();
    T* RU = _RU.data();
    size_t n = _order+1;
    size_t nsys = this->Nsys();
    compute_R(_R.data(), _order, factor);
    compute_R(_U.data(), _order, T(1));

    for (size_t i=0; i<_RU.size(); i++){
        RU[i]=0;
    }

    for (size_t i=0; i<n; i++){
        for (size_t k=0; k<n; k++){
            T p =  R[i*n+k];
            #pragma omp simd
            for (size_t j=0; j<n; j++){
                RU[i * n+j] += p*U[k*n+j];
            }
        }
    }

    Dlike& D_new = _D[1-_idx_D];
    const Dlike& D = _D[_idx_D];

    for (size_t i=0; i<n; i++){
        for (size_t j=0; j< nsys; j++){
            T sum = 0;
            for (size_t k=0; k<n; k++){
                sum += RU[k*n+i]*D[k*nsys+j];
            }
            D_new[i*nsys+j] = sum;
        }
    }

    _idx_D = 1 - _idx_D;
    _n_eq_steps = 0;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::_set_prediction(T* y){
    size_t n=this->Nsys();
    T* D = _D[_idx_D].data();

    for (size_t j=0; j < n; j++){
        y[j] = 0;
    }

    for (size_t i=0; i < _order+1; i++){
        #pragma omp simd
        for (size_t j=0; j < n; j++){
            y[j] += D[i * n + j];
        }
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BDF<T, N, SP, RhsType, JacType>::_set_psi(T* psi){
    size_t n = this->Nsys();
    const T* D = _D[_idx_D].data();
    const T* g = BDF_COEFS.GAMMA.data();
    const T& a = BDF_COEFS.ALPHA[_order];

    for (size_t j=0; j < n; j++){
        psi[j] = 0;
    }
    for (size_t i=1; i<_order+1; i++){
        //optimize
        #pragma omp simd
        for (size_t j=0; j<n; j++){
            psi[j] += D[i * n + j] * g[i] / a;
        }
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BDF<T, N, SP, RhsType, JacType>::_resize_step(T& factor, const T& min_step, const T& max_step){
    //factor should be positive
    bool res = resize_step(factor, min_step, max_step); //automatically changes factor if needed
    _change_D(factor);
    return res;
}

} // namespace ode

#endif
