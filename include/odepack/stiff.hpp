#ifndef STIFF_HPP
#define STIFF_HPP

#include "solver_impl.hpp"
#include <stdexcept>

const int MAX_ORDER = 5;

template<typename T, int N>
using LUResult = Eigen::PartialPivLU<JacMat<T, N>>;

template<typename T>
vec<T> arange(const int& a, const int& b);

template<typename T, int N>
vec<T, N> cumsum(const vec<T, N>& x);

template<typename T, int N>
void cumprod(Eigen::Matrix<T, -1, 1>& res, const vec<T, N>& x);

template<typename T>
Eigen::Matrix<T, -1, -1> compute_R(const int& order, const T& factor);

template<typename T, int N>
Eigen::Matrix<T, N, -1> solve_lu(const LUResult<T, N>& lu_decomp, const vec<T, N>& b);

template<typename T>
struct BDFCONSTS{

    vec<T, MAX_ORDER+1> KAPPA;
    vec<T, MAX_ORDER+1> GAMMA;
    vec<T, MAX_ORDER+1> ALPHA;
    vec<T, MAX_ORDER+1> ERR_CONST;

    BDFCONSTS();

};



template<typename T, int N>
struct _MutableBDF{

    _MutableBDF(const vec<T, N>& q);

    vec<T, N> ypredict;
    vec<T, N> scale;
    vec<T, N> psi;
    vec<T, N> dy;
    vec<T, N> f;
    vec<T, N> error_m, error_p;
    vec<T, 3> error_norms, factors;

};

struct NewtConv{
    bool converged;
    int n_iter;
};


template<typename T, int N>
class BDF;


template<typename T, int N>
class BDFInterpolator;

template<typename T, int N>
class StateBDF final: public DerivedState<T, N, StateBDF<T, N>>{

    friend class BDF<T, N>;

    friend class BDFInterpolator<T, N>;
    
    static const BDFCONSTS<T> bdf;

    using Base = DerivedState<T, N, StateBDF<T, N>>;

public:

    StateBDF(const T& t, const vec<T, N>& q, const T& h);

    DEFAULT_RULE_OF_FOUR(StateBDF);

    StateBDF() = delete;
    
    bool resize_step(T& factor, const T& min_step=0, const T& max_step=inf<T>()) final;

    void adjust(const T& h_abs, const T& dir, const vec<T, N>& diff) final;

    const vec<T, N>& q_predict()const;

    const vec<T, N>& psi()const;

    void set_LU();
    
private:

    void _change_D(const T& factor);

    JacMat<T, N> I;
    JacMat<T, N> J;
    Eigen::Matrix<T, N, MAX_ORDER+3> D;
    LUResult<T, N> LU;
    int n_eq_steps = 0;
    int order=1;
    bool lu=false;
    mutable vec<T, N> _q_predict;
    mutable bool _pred_is_stored = false;
    mutable vec<T, N> _psi;
    mutable bool _psi_is_stored = false;
    vec<T, N> d;

};



template<typename T, int N>
class BDFInterpolator final : public LocalInterpolator<T, N>{

public:
    
    BDFInterpolator() = delete;

    DEFAULT_RULE_OF_FOUR(BDFInterpolator);

    BDFInterpolator(const T& t, const vec<T, N>& q);

    BDFInterpolator(const StateBDF<T, N>& state1, const StateBDF<T, N>& state2, int bdr1, int bdr2);

    int order() const final;

    BDFInterpolator<T, N>* clone() const override;


private:

    void _call_impl(vec<T, N>& result, const T& t) const final;

    int _order = 1;
    vec<T> _t_shift;
    vec<T> _denom;
    Eigen::Matrix<T, N, -1> _D;

    mutable Eigen::Matrix<T, -1, 1> _p;

};


template<typename T, int N>
class BDF : public DerivedSolver<T, N, BDF<T, N>, StateBDF<T, N>, BDFInterpolator<T, N>>{

    using STATE = StateBDF<T, N>;
    using SolverBase = DerivedSolver<T, N, BDF<T, N>, StateBDF<T, N>, BDFInterpolator<T, N>>;

    friend class DerivedSolver<T, N, BDF<T, N>, StateBDF<T, N>, BDFInterpolator<T, N>>;

public:

    static const int NEWTON_MAXITER = 4;
    static const int ERR_EST_ORDER = 1; //the actual order flactuates

    BDF(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(BDF);

    OdeRhs<T, N> ode_rhs() const final;

    STATE new_state(const T& t, const vec<T, N>& q, const T& h) const;

    inline BDFInterpolator<T, N> state_interpolator(const STATE& state1, const STATE& state2, int bdr1, int bdr2) const;

    void adapt_impl(STATE& res, const STATE& state);

private:

    inline void _jac(JacMat<T, N>& result, const T& t, const vec<T, N>& q) const;

    NewtConv _solve_bdf_system(StateBDF<T, N>& res, const T& t_new, const vec<T, N>& scale);

    void _reset_impl(){
        _mut = _MutableBDF<T, N>(this->q());
        
    }

    T _newton_tol;
    Jac<T, N> _jacobian;
    mutable _MutableBDF<T, N> _mut;

};


/*
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----------------------------------------IMPLEMENTATIONS-------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
*/



template<typename T, int N>
const BDFCONSTS<T> StateBDF<T, N>::bdf = BDFCONSTS<T>();


template<typename T>
vec<T> arange(const int& a, const int& b){
    vec<T> res(b-a);
    for (int i=0; i<(b-a); i++){
        res[i] = a+i;
    }
    return res;
}

template<typename T, int N>
vec<T, N> cumsum(const vec<T, N>& x){
    vec<T> res(x.size());
    res[0] = x[0];
    for (int i=1; i<x.size(); i++){
        res[i] = res[i-1]+x[i];
    }
    return res;
}

template<typename T, int N>
void cumprod(Eigen::Matrix<T, -1, 1>& res, const vec<T, N>& x){
    res[0] = x[0];
    for (int i=1; i<x.size(); i++){
        res[i] = res[i-1]*x[i];
    }
}

template<typename T>
Eigen::Matrix<T, -1, -1> compute_R(const int& order, const T& factor) {
    using MatrixXT = Eigen::Matrix<T, -1, -1>;

    MatrixXT M = MatrixXT::Zero(order + 1, order + 1);

    for (int i = 1; i <= order; ++i) {
        for (int j = 1; j <= order; ++j) {
            M(i, j) = (i-1 - factor * j) / i;
        }
    }

    M.row(0).setOnes();

    // Cumulative product along axis 0 (down the columns)
    MatrixXT R = M;
    for (int i = 1; i <= order; ++i) {
        R.row(i) = R.row(i - 1).cwiseProduct(M.row(i));
    }

    return R;
}

template<typename T, int N>
Eigen::Matrix<T, N, -1> solve_lu(const LUResult<T, N>& lu_decomp, const vec<T, N>& b){
    return lu_decomp.solve(b.matrix());
}


template<typename T>
BDFCONSTS<T>::BDFCONSTS(){
    KAPPA = {0, -T(185)/1000, -T(1)/9, -T(823)/10000, -T(415)/10000, 0};
    GAMMA[0] = 0;
    T cumulative = 0;
    for (int i = 1; i < MAX_ORDER+1; ++i) {
        cumulative += T(1) / i;
        GAMMA[i] = cumulative;
    }
    ALPHA = ((1-KAPPA)*GAMMA).eval();
    ERR_CONST = (KAPPA * GAMMA + 1/arange<T>(1, MAX_ORDER+2)).eval();
}

template<typename T, int N>
_MutableBDF<T, N>::_MutableBDF(const vec<T, N>& q):ypredict(q), scale(q), psi(q), dy(q), f(q), error_m(q), error_p(q){}


template<typename T, int N>
StateBDF<T, N>::StateBDF(const T& t, const vec<T, N>& q, const T& h) : DerivedState<T, N, StateBDF<T, N>>(t, q, h){
    //state not fully initialized yet. It needs to be "adjusted".
    I = Eigen::Matrix<T, N, N>::Identity(q.size(), q.size());
    _q_predict.resize(q.size());
    _psi.resize(q.size());
    d.resize(q.size());
    d = 0;
    D.resize(q.size(), MAX_ORDER+3);
    J.resize(q.size(), q.size());
    J.setZero();
}


template<typename T, int N>
bool StateBDF<T, N>::resize_step(T& factor, const T& min_step, const T& max_step){
    //factor should be positive
    bool res = Base::resize_step(factor, min_step, max_step); //automatically changes factor if needed
    _change_D(factor);
    n_eq_steps = 0;
    return res;
}

template<typename T, int N>
void StateBDF<T, N>::adjust(const T& h_abs, const T& dir, const vec<T, N>& diff){
    Base::adjust(h_abs, dir, diff);
    D.setZero();
    D.col(0) = this->_q;
    D.col(1) = this->h()*diff; //dq
    n_eq_steps = 0;
    order = 1;
    lu = false;
    _pred_is_stored = false;
    _psi_is_stored = false;
}

template<typename T, int N>
const vec<T, N>& StateBDF<T, N>::q_predict()const{
    if (!_pred_is_stored){
        _q_predict = D.leftCols(order + 1).rowwise().sum();
        _pred_is_stored = true;
    }
    return _q_predict;
}

template<typename T, int N>
const vec<T, N>& StateBDF<T, N>::psi() const {
    if (!_psi_is_stored){
        _psi = ((D.middleCols(1, order) * bdf.GAMMA.segment(1, order).matrix()) / bdf.ALPHA[order]);
        _psi_is_stored = true;
    }
    return _psi;
}

template<typename T, int N>
void StateBDF<T, N>::set_LU(){
    JacMat<T, N> A = (I - this->h() / bdf.ALPHA[order] * J).eval();
    LU = Eigen::PartialPivLU<JacMat<T, N>>(A);
    lu = true;
}


template<typename T, int N>
void StateBDF<T, N>::_change_D(const T& factor){
    // Compute transformation matrices
    Eigen::Matrix<T, -1, -1> R = compute_R<T>(order, factor);
    Eigen::Matrix<T, -1, -1> U = compute_R<T>(order, 1);
    Eigen::Matrix<T, -1, -1> RU = R * U;

    Eigen::Matrix<T, -1, -1> D_sub = D.leftCols(order + 1);
    D_sub = RU.transpose() * D_sub.transpose();

    D.leftCols(order + 1) = D_sub.transpose();
    _pred_is_stored = false;
    _psi_is_stored = false;
}


template<typename T, int N>
BDF<T, N>::BDF(MAIN_CONSTRUCTOR(T, N)) : SolverBase("BDF", ARGS), _jacobian(rhs.jacobian), _mut(q0){    
    if (_jacobian == nullptr){
        throw std::runtime_error("Please provide the Jacobian matrix function of the ODE system when using the BDF method");
    }
    if (rtol == 0){
        rtol = 100*std::numeric_limits<T>::epsilon();
        std::cerr << "Warning: rtol=0 not allowed in the BDF method. Setting rtol = " << rtol;
    }
    _newton_tol = std::max(10 * std::numeric_limits<T>::epsilon() / rtol, std::min(T(3)/100, pow(rtol, T(1)/T(2))));

    this->_finalize(t0, q0, first_step);
}

template<typename T, int N>
OdeRhs<T, N> BDF<T, N>::ode_rhs() const {
    return {this->_rhs(), this->_jacobian};
}

template<typename T, int N>
StateBDF<T, N> BDF<T, N>::new_state(const T& t, const vec<T, N>& q, const T& h) const{
    STATE res = {t, q, h};
    res.adjust(res._habs, sgn(h), this->_rhs(t, q));
    this->_jac(res.J, t, q);
    return res;
}

template<typename T, int N>
inline BDFInterpolator<T, N> BDF<T, N>::state_interpolator(const STATE& state1, const STATE& state2, int bdr1, int bdr2) const{
    return BDFInterpolator<T, N>(state1, state2, bdr1, bdr2);
}

template<typename T, int N>
void BDF<T, N>::adapt_impl(STATE& res, const STATE& state){
    const T& t = this->t();
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const T& atol = this->atol();
    const T& rtol = this->rtol();

    res = state; //ideally we would avoid this copy
    JacMat<T, N>& J = res.J;
    T& t_new = res._t;
    T safety, error_norm, error_m_norm, error_p_norm, max_factor, factor;
    int& order = res.order;
    bool converged;
    bool current_jac = false;
    NewtConv conv_result;
    bool step_accepted = false;
    while (!step_accepted){
        
        t_new = t + res.h();

        _mut.scale = atol + rtol * res.q_predict().cwiseAbs();
        converged = false;
        while (!converged){
            if (!res.lu){
                res.set_LU();
            }
            conv_result = _solve_bdf_system(res, t_new, _mut.scale);
            converged = conv_result.converged;
            if (!converged){
                if (current_jac) break;
                this->_jac(J, t_new, res.q_predict());
                res.lu = false;
                current_jac = true;
            }
        }

        if (!converged){
            T factor = T(1)/T(2);
            if (res.resize_step(factor, h_min, max_step)){
                res.lu = false;
                continue;
            }
            else{
                res.D.col(order+2) = res.d.matrix() - res.D.col(order+1);
                res.D.col(order+1) = res.d;
                for (int i = order; i >= 0; --i) {
                    res.D.col(i) += res.D.col(i+1);
                }
                res.lu = false; //not sure if this is important
                return;
            }
        }

        safety = T(9)/T(10) * T(2*NEWTON_MAXITER+1)/T(2*NEWTON_MAXITER+conv_result.n_iter);
        _mut.scale = atol + rtol * res.vector().cwiseAbs();
        error_norm = rms_norm((res._error/_mut.scale).eval());

        if (error_norm > 1){
            factor = std::max(this->MIN_FACTOR, safety * pow(error_norm, T(-1)/(order+1)));
            if (res.resize_step(factor, h_min, max_step)){
                continue;
            }
            else{
                res.D.col(order+2) = res.d.matrix() - res.D.col(order+1);
                res.D.col(order+1) = res.d;
                for (int i = order; i >= 0; --i) {
                    res.D.col(i) += res.D.col(i+1);
                }
                res.lu = false;
                return;
            }
            //no LU resetting
        }
        else{
            step_accepted = true;
        }
    }

    
    res.n_eq_steps++;
    // res.set_LU();

    res.D.col(order+2) = res.d.matrix() - res.D.col(order+1);
    res.D.col(order+1) = res.d;
    for (int i = order; i >= 0; --i) {
        res.D.col(i) += res.D.col(i+1);
    }
    res._pred_is_stored = false;
    res._psi_is_stored = false;
    if (res.n_eq_steps < order+1){
        return;
    }
    if (order>1){
        _mut.error_m = res.bdf.ERR_CONST[order-1] * res.D.col(order);
        error_m_norm = rms_norm((_mut.error_m/_mut.scale).eval());
    }
    else{
        error_m_norm = inf<T>();
    }

    if (order < MAX_ORDER){
        _mut.error_p = res.bdf.ERR_CONST[order+1] * res.D.col(order+2);
        error_p_norm = rms_norm((_mut.error_p/_mut.scale).eval());
    }
    else{
        error_p_norm = inf<T>();
    }
    _mut.error_norms = {error_m_norm, error_norm, error_p_norm};
    _mut.factors = pow(_mut.error_norms, -1/vec<T, 3>({static_cast<T>(order), static_cast<T>(order+1), static_cast<T>(order+2)}));

    Eigen::Index maxIndex;
    max_factor = _mut.factors.maxCoeff(&maxIndex);
    order += maxIndex-1;
    factor = std::min(this->MAX_FACTOR, safety*max_factor);
    res.resize_step(factor, h_min, max_step);
    res.lu = false;
}

template<typename T, int N>
inline void BDF<T, N>::_jac(JacMat<T, N>& result, const T& t, const vec<T, N>& q) const{
    _jacobian(result, t, q, this->args());
}

template<typename T, int N>
NewtConv BDF<T, N>::_solve_bdf_system(StateBDF<T, N>& res, const T& t_new, const vec<T, N>& scale) {
    res.d = 0;
    res._q = res.q_predict();

    T dy_norm = 0;
    T dy_norm_old = 0;
    T rate = 0;
    T c = res.h()/res.bdf.ALPHA[res.order];
    bool converged = false;
    int j = 0;
    for (int k=0; k<NEWTON_MAXITER; k++){
        this->_rhs(_mut.f, t_new, res._q);
        if (!_mut.f.isFinite().all()){
            break;
        }
        _mut.dy = solve_lu(res.LU, (c*_mut.f-res.psi()-res.d).eval()).array();

        dy_norm = rms_norm((_mut.dy/scale).eval());
        if (dy_norm_old == 0){
            rate = 0;
        }
        else{
            rate = dy_norm/dy_norm_old;
        }

        if (rate != 0 && (rate >= 1 || (pow(rate, NEWTON_MAXITER-k)/(1-rate)*dy_norm>_newton_tol))){
            break;
        }

        res._q += _mut.dy;
        res.d += _mut.dy;
        if (dy_norm == 0 || ((rate != 0) && (rate/(1-rate)*dy_norm<_newton_tol))){
            converged = true;
            break;
        }

        dy_norm_old = dy_norm;
        j++;
    }
    res._error = res.bdf.ERR_CONST[res.order] * res.d;
    return {converged, j+1};
}

template<typename T, int N>
BDFInterpolator<T, N>::BDFInterpolator(const T& t, const vec<T, N>& q) : LocalInterpolator<T, N>(t, q) {}


template<typename T, int N>
BDFInterpolator<T, N>::BDFInterpolator(const StateBDF<T, N>& state1, const StateBDF<T, N>& state2, int bdr1, int bdr2) : LocalInterpolator<T, N>(state1, state2, bdr1, bdr2), _order(state2.order), _t_shift(state2.t()-state2.h()*arange<T>(0, state2.order)), _denom(state2.h()*(1+arange<T>(0, state2.order))), _D(state2.D.leftCols(state2.order+1)), _p(state2.order, 1) {}

template<typename T, int N>
int BDFInterpolator<T, N>::order() const {
    return _order;
}


template<typename T, int N>
BDFInterpolator<T, N>* BDFInterpolator<T, N>::clone() const {
    return new BDFInterpolator<T, N>(*this);
}


template<typename T, int N>
void BDFInterpolator<T, N>::_call_impl(vec<T, N>& result, const T& t) const {
    cumprod(_p, ((t-_t_shift)/_denom).eval());
    result = (_D.col(0) + _D.rightCols(_order) * _p).array();
}
#endif