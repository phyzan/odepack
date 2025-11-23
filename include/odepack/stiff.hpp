#ifndef STIFF_HPP
#define STIFF_HPP

#include "solver_impl.hpp"


constexpr size_t MAX_ORDER = 5;

template<typename T, size_t N>
struct LUResult {

    LUResult(size_t Nsys) : LU(Nsys, Nsys), piv(Nsys) {}

    Array2D<T, N, N> LU; 
    Array1D<size_t, N> piv;
};

template<typename T, size_t N>
void lu_factor(LUResult<T, N>& res, const Array2D<T, N, N>& A_input) {
    size_t n = A_input.Nrows();
    Array2D<T, N, N>& A = res.LU;
    Array1D<size_t, N>& piv = res.piv;
    
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
void lu_solve(T* x, const LUResult<T, N>& LU_struct, const T* b) {
    const Array2D<T, N, N>& LU = LU_struct.LU;
    const Array1D<size_t, N>& piv = LU_struct.piv;
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
struct BDFCONSTS{

    Array1D<T, MAX_ORDER+1> KAPPA;
    Array1D<T, MAX_ORDER+1> GAMMA;
    Array1D<T, MAX_ORDER+1> ALPHA;
    Array1D<T, MAX_ORDER+1> ERR_CONST;

    BDFCONSTS(){
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
};


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
class BDFInterpolator final : public LocalInterpolator<T, N>{

public:
    
    BDFInterpolator() = delete;

    DEFAULT_RULE_OF_FOUR(BDFInterpolator);

    BDFInterpolator(const T& t, const Array1D<T, N>& q) : LocalInterpolator<T, N>(t, q) {}

    BDFInterpolator(const Array2D<T, 0, N>& D, size_t order, const State<T, N>& state1, const State<T, N>& state2, int bdr1, int bdr2) : LocalInterpolator<T, N>(state1.t, state2.t, state1.vector, state2.vector, bdr1, bdr2), _order(order), _D(order+1, D.Ncols()) {
        int dir = sgn(state2.t - state1.t);
        _h = state2.habs*dir;
        _t2 = state2.t;
        copy_array(_D.data(), D.data(), _D.size()); //keeping the first (order+1) rows
    }

    size_t order() const final{
        return _order;
    }

    BDFInterpolator<T, N>* clone() const final{
        return new BDFInterpolator<T, N>(*this);
    }


private:

    void _call_impl(T* result, const T& t) const final{
        bdf_interp(result, t, _t2, _h, _D.data(), _order, this->array_size());
    }

    size_t _order = 1;
    T _h;
    T _t2;
    Array2D<T, 0, N> _D;

};

struct NewtConv{
    bool converged;
    size_t n_iter;
};



template<typename T, size_t N>
class BDF : public DerivedSolver<T, N, BDF<T, N>>{

    using Base = DerivedSolver<T, N, BDF<T, N>>;
    using Dlike = Array2D<T, 0, N>;

public:

    static constexpr size_t NEWTON_MAXITER = 4;
    static constexpr size_t ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = true;

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T, N)) : Base("BDF", ARGS), _J(q0.size(), q0.size()), _B(q0.size(), q0.size()), _LU(q0.size()), _R((MAX_ORDER+1)*(MAX_ORDER+1)), _U((MAX_ORDER+1)*(MAX_ORDER+1)), _RU((MAX_ORDER+1)*(MAX_ORDER+1)), _f(q0.size()), _dy(q0.size()), _b(q0.size()), _scale(q0.size()), _ypred(q0.size()), _psi(q0.size()), _d(q0.size()), _error(q0.size()), _error_m(q0.size()), _error_p(q0.size()) {
        if (ode.jacobian == nullptr){
            throw std::runtime_error("Please provide the Jacobian matrix function of the ODE system when using the BDF method");
        }
        if (rtol == 0){
            rtol = 100*std::numeric_limits<T>::epsilon();
            std::cerr << "Warning: rtol=0 not allowed in the BDF method. Setting rtol = " << rtol << std::endl;
        }
        _newton_tol = std::max(10 * std::numeric_limits<T>::epsilon() / rtol, std::min(T(3)/100, pow(rtol, T(1)/T(2))));
        
        
        this->reset();

    }

    DEFAULT_RULE_OF_FOUR(BDF)

    void re_adjust() {
        copy_array(_D[0].data(), this->current_state().vector.data(), this->Nsys());
        this->_rhs(_D[0].data()+this->Nsys(), this->t(), this->current_state().vector.data());
        for (size_t i=0; i<this->Nsys(); i++){
            _D[0][i+this->Nsys()] *= this->stepsize() * this->direction();
        }
        this->_jac(_J, this->t(), this->current_state().vector);
        _order = 1;
        _n_eq_steps = 0;
        _valid_LU = false;
        _idx_D = 0;
    }

    void reset() override{
        T t0 = this->ics().t;
        T h0 = this->ics().habs * this->direction();
        const Array1D<T, N>& q0 = this->ics().vector;
        Array1D<T, N> f(this->Nsys());
        this->_rhs(f.data(), t0, q0.data());

        for (size_t k = 0; k<2; k++){
            _D[k] = Dlike(MAX_ORDER+3, this->Nsys());
            for (size_t j=0; j<this->Nsys(); j++){
                _D[k](0, j) = q0[j];
                _D[k](1, j) = f[j] * h0;
            }
        }
        this->_jac(_J, t0, q0);
        _order = 1;
        _n_eq_steps = 0;
        _valid_LU = false;
        _idx_D = 0;
    }

    void adapt_impl(State<T, N>& res){
        const T& h_min = this->min_step();
        const T& max_step = this->max_step();
        const T& atol = this->atol();
        const T& rtol = this->rtol();
        const State<T, N>& state = this->current_state();
        const T& t = state.t;
        size_t nsys = this->Nsys();


        res.t = state.t;
        copy_array(res.vector.data(), state.vector.data(), nsys);

        T& t_new = res.t;
        T* y_new = res.vector.data();
        T safety, max_factor, factor, c;
        int delta_order;
        bool converged;
        bool current_jac = false;
        bool step_accepted = false;
        NewtConv conv_result;

        if (state.habs > max_step){
            res.habs = max_step;
            _change_D(max_step/state.habs);
        }
        else if (state.habs < h_min){
            res.habs = h_min;
            _change_D(h_min/state.habs);
        }
        else{
            res.habs = state.habs;
        }
        

        while(!step_accepted){
            if (res.habs < h_min){
                this->kill("The stepsize for the BDF method dropped below min_step");
                return;
            }
            else if (res.habs > max_step){
                _change_D(max_step/res.habs);
                res.habs = max_step;
                return;
            }

            t_new = t + res.habs * this->direction();

            _set_prediction(_ypred.data());
            #pragma omp simd
            for (size_t i=0; i<nsys; i++){
                _scale[i] = atol + rtol * abs(_ypred[i]);
            }
            _set_psi(_psi.data());

            converged = false;
            c = res.habs * this->direction() / BDF_COEFS.ALPHA[_order];
            while (!converged){
                if (!_valid_LU){
                    for (size_t i=0; i<nsys; i++){
                        for (size_t j=0; j<nsys; j++){
                            if (i == j){
                                _B(i, j) = 1 - c*_J(i, j);
                            }
                            else{
                                _B(i, j) = -c*_J(i, j);
                            }
                        }
                    }
                    lu_factor(_LU, _B);
                    _valid_LU = true;
                }

                conv_result = _solve_bdf_system(y_new, _ypred.data(), _d, t_new, c, _psi, _LU, _scale);
                converged = conv_result.converged;

                if (!converged){
                    if (current_jac){
                        break;
                    }
                    this->_jac(_J, t_new, _ypred);
                    _valid_LU = false;
                    current_jac = true;
                }
            }

            if (!converged){
                factor = T(1)/2;
                res.habs *= factor;
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
                res.habs *= factor;
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
        res.habs *= factor;
        _change_D(factor);


        _valid_LU = false;
    }

    inline std::unique_ptr<Interpolator<T, N>> state_interpolator(int bdr1, int bdr2) const{
        return std::unique_ptr<Interpolator<T, N>>(new BDFInterpolator<T, N>(_D[_idx_D], _order, this->old_state(), this->current_state(), bdr1, bdr2));
    }

    inline void interp_impl(T* result, const T& t) const{
        bdf_interp(result, t, this->t(), this->current_state().habs*this->direction(), _D[_idx_D].data(), _order, this->Nsys());
    }


private:

    NewtConv _solve_bdf_system(T* y, const T* y_pred, Array1D<T, N>& d, const T& t_new, const T& c, const Array1D<T, N>& psi, const LUResult<T, N>& LU, const Array1D<T, N>& scale){
        d.set(0);
        size_t n = this->Nsys();
        copy_array(y, y_pred, n);
        T dy_norm = 0;
        T dy_norm_old = 0;
        T rate = 0;
        bool converged = false;
        size_t j=0;
        for (size_t k=0; k<NEWTON_MAXITER; k++){
            this->_rhs(_f.data(), t_new, y);
            if (!all_are_finite(_f.data(), n)){
                break;
            }

            #pragma omp simd
            for (size_t i=0; i<n; i++){
                _b[i] = c * _f[i] - psi[i] - d[i];
            }

            lu_solve(_dy.data(), LU, _b.data());
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

    void _change_D(const T& factor){
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

    void _set_prediction(T* y){
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

    void _set_psi(T* psi){
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

    bool _resize_step(T& factor, const T& min_step, const T& max_step){
        //factor should be positive
        bool res = resize_step(factor, min_step, max_step); //automatically changes factor if needed
        _change_D(factor);
        return res;
    }

    JacMat<T, N> _J;
    mutable JacMat<T, N> _B;
    std::array<Dlike, 2> _D;
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

};



#endif