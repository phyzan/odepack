#ifndef STIFF_HPP
#define STIFF_HPP

#include "odesolvers.hpp"
#include <memory>

const int MAX_ORDER = 5;

template<class T, int N>
using LUResult = Eigen::PartialPivLU<JacMat<T, N>>;

template<class T, int N>
LUResult<T, N> lu(const JacMat<T, N>& A) {
    return LUResult(A);
}

// Solving Ax = b using LU decomposition
template<class T>
vec<T> arange(int a, int b){
    vec<T> res(b-a);
    for (int i=0; i<(b-a); i++){
        res[i] = a+i;
    }
    return res;
}

template<class T, int N>
vec<T, N> cumsum(const vec<T, N>& x){
    vec<T> res(x.size());
    res[0] = x[0];
    for (int i=1; i<x.size(); i++){
        res[i] = res[i-1]+x[i];
    }
    return res;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
compute_R(const int& order, const T& factor) {
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

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

template<class T, int N>
Eigen::Matrix<T, N, -1> solve_lu(const LUResult<T, N>& lu_decomp, const vec<T, N>& b){
    return lu_decomp.solve(b.matrix());
}

template<class T>
struct BDFCONSTS{

    vec<T, MAX_ORDER+1> KAPPA;
    vec<T, MAX_ORDER+1> GAMMA;
    vec<T, MAX_ORDER+1> ALPHA;
    vec<T, MAX_ORDER+1> ERR_CONST;

    BDFCONSTS(){
        KAPPA = {0, -T(185)/1000, -T(1)/9, -T(823)/10000, -T(415)/10000, 0};
        GAMMA[0] = 0;
        T cumulative = 0;
        for (int i = 1; i < MAX_ORDER+1; ++i) {
            cumulative += 1 / i;
            GAMMA[i] = cumulative;
        }
        ALPHA = ((1-KAPPA)*GAMMA).eval();
        ERR_CONST = (KAPPA * GAMMA + 1/arange<T>(1, MAX_ORDER+2)).eval();
    }

};



template<class T, int N>
struct _MutableBDF{

    _MutableBDF(const vec<T, N>& q):ypredict(q), scale(q), psi(q), dy(q), f(q), error_m(q), error_p(q){
        J.resize(q.size(), q.size());
    }

    JacMat<T, N> J;
    vec<T, N> ypredict;
    vec<T, N> scale;
    vec<T, N> psi;
    vec<T, N> dy;
    vec<T, N> f;
    vec<T, N> error_m, error_p;
    vec<T, 3> error_norms, factors;

};


template<class T, int N>
class BDF;

template<class T, int N>
class StateBDF final: public DerivedState<T, N, StateBDF<T, N>>{

    //when a state is initialized, direction has not been set.
    //So values inside matrices are still empty, and when more information is provided
    //the state will be fully created.
    friend class BDF<T, N>;
    
    static const BDFCONSTS<T> bdf;

public:

    StateBDF(const T& t, const vec<T, N>& q, const T& habs) : DerivedState<T, N, StateBDF<T, N>>(t, q, habs){
        I = Eigen::Matrix<T, N, N>::Identity(q.size(), q.size());
        _q_predict.resize(q.size());
        _psi.resize(q.size());
        d.resize(q.size());
        d = 0;
        D.resize(q.size(), MAX_ORDER+3);
    }

    StateBDF() = delete;

    bool resize_step(T factor, const T& min_step=0, const T& max_step=inf<T>()){
        //factor should be positive
        bool res = State<T, N>::resize_step(factor, min_step, max_step); //automatically changes factor if needed
        _change_D(factor);
        n_eq_steps = 0;
        return res;
    }

    void adjust(const T& h_abs, const T& dir, const vec<T, N>& diff)final{
        State<T, N>::adjust(h_abs, dir, diff);
        D.setZero();
        D.col(0) = this->_q;
        D.col(1) = h_abs*this->_direction*diff; //dq
        n_eq_steps = 0;
        order = 1;
        lu = false;
        _pred_is_stored = false;
        _psi_is_stored = false;
    }

    void assign(const T& t, const vec<T, N>& q, const T& habs, const T& dir, const vec<T, N>& diff) final{
        this->_t = t;
        this->_q = q;
        adjust(habs, dir, diff);
    }

    const vec<T, N>& q_predict()const{
        if (!_pred_is_stored){
            _q_predict = D.leftCols(order + 1).rowwise().sum();
            _pred_is_stored = true;
        }
        return _q_predict;
    }

    const vec<T, N>& psi()const{
        if (!_psi_is_stored){
            _psi = ((D.middleCols(1, order) * GAMMA.segment(1, order).matrix()) / ALPHA[order]);
            _psi_is_stored = true;
        }
        return _psi;
    }

    void set_LU(const JacMat<T, N>& J){
        JacMat<T, N> A = (I - this->h() / bdf.ALPHA[order] * J).eval();
        LU = Eigen::PartialPivLU<JacMat<T, N>>(A);
        lu = true;
    }
    
private:

    JacMat<T, N> I;
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



    void _change_D(const T& factor){
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

    void finalize()

};


struct NewtConv{
    bool converged;
    int n_iter;
};


template<class T, int N>
class BDF : public OdeSolver<T, N>{

    using STATE = StateBDF<T, N>;

public:

    const int NEWTON_MAXITER = 4;

    BDF(Jac<T, N> jac, MAIN_DEFAULT_CONSTRUCTOR(T, N)) : DerivedSolver<T, N, StateBDF<T, N>>(rhs, rtol, atol, min_step, max_step, args, events, mask, save_dir, save_events_only, 1, new StateBDF<T, N>(t0, q0, first_step)), _jacobian(jac), _mut(q0){

        _newton_tol = std::max(10 * std::numeric_limits<T>::epsilon() / rtol, std::min(T(3)/100, pow(rtol, T(1)/T(2))));
        
        _jac(_mut.J, t0, q0);
    }

    OdeRhs<T, N> ode_rhs() const final{
        return {this->_ode_rhs, this->_jacobian};
    }

    OdeSolver<T, N>* clone() const override{
        return new BDF<T, N>(*this);
    }

    std::unique_ptr<OdeSolver<T, N>> safe_clone() const override{
        return std::make_unique<BDF<T, N>>(*this);
    }

private:

    void _step_impl(STATE& result, const STATE& state, const T& h) const final{
        result = state;
        _mut.scale = this->atol() + this->rtol() * state.q_predict().cwiseAbs();
        _solve_bdf_system(result, state.t+h, _mut.scale);
    }

    void _adapt_impl()final{
        const T& t = this->t();
        const T& h_min = this->min_step();
        const T& max_step = this->max_step();
        const T& atol = this->atol();
        const T& rtol = this->rtol();

        
        STATE& res = *this->_old_state_ptr();
        const STATE& state = *this._state_ptr();
        res = state; //ideally we would avoid this copy
        res.lu = false;

        JacMat<T, N>& J = _mut.J;
        T& t_new = res._t;
        T safety, error_norm, error_m_norm, error_p_norm, max_factor, factor;
        int& order = res.order;
        bool converged;
        int n_iter;
        bool current_jac = false;
        NewtConv conv_result;
        bool step_accepted = false;
        while (!step_accepted){            
            t_new = t + res.h();

            _mut.scale = atol + rtol * res.q_predict().cwiseAbs();
            converged = false;
            while (!converged){
                if (!res.lu){
                    res.set_LU(J);
                }
                conv_result = _solve_bdf_system(res, t_new, _mut.scale);
                converged = conv_result.converged;
                n_iter = conv_result.n_iter;
                if (!converged){
                    if (current_jac) break;
                    this->_jac(J, t_new, res.q_predict());
                    res.lu = false;
                    current_jac = true;
                }
            }
            if (!converged){
                if (res.resize_step(T(1)/T(2), h_min, max_step)){
                    res.lu = false;
                    continue;
                }
                else{
                    res.D.col(order+2) = res.d.matrix() - res.D.col(order+1);
                    res.D.col(order+1) = res.d;
                    for (int i = order; i >= 0; --i) {
                        res.D.col(i) += res.D.col(i+1);
                    }
                    res.lu = false;
                    // state.set_LU(this->_jac(t_new, q));
                    return;
                }
            }

            safety = T(9)/T(10) * T(2*NEWTON_MAXITER+1)/T(2*NEWTON_MAXITER+n_iter);
            _mut.scale = atol + rtol * res.q.cwiseAbs();
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
                    // state.set_LU(this->_jac(t_new, q));
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
        res.set_LU(_mut.J);

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
        _mut.factors = pow(error_norms, -1/vec<T, 3>({static_cast<T>(order), static_cast<T>(order+1), static_cast<T>(order+2)}));

        Eigen::Index maxIndex;
        max_factor = factors.maxCoeff(&maxIndex);
        order += maxIndex-1;
        factor = std::min(this->MAX_FACTOR, safety*max_factor);
        res.resize_step(factor, h_min, max_step);
        // state.set_LU(this->_jac(t_new, q));
        res.lu = false;
    }

    inline void _jac(JacMat<T, N>& result, const T& t, const vec<T, N>& q){
        _jacobian(result, t, q, this->args());
    }

    const JacMat<T, N>& _jac(const T& t, const vec<T, N>& q)const{
        _jacobian(_mut.J, t, q, this->args());
        return _mut.J;
    }

    NewtConv _solve_bdf_system(StateBDF<T, N>& res, const T& t_new, const vec<T, N>& scale) {
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

            if (dy_norm == 0 || (rate != 0 && rate/(1-rate)*dy_norm<_newton_tol)){
                converged = true;
                break;
            }

            dy_norm_old = dy_norm;
            j++;
        }
        res._error = res.bdf.ERR_CONST[res.order] * res.d;
        return {converged, j+1};
    }
    

    T _newton_tol;
    Jac<T, N> _jacobian;
    mutable _MutableBDF<T, N> _mut;

};


#endif