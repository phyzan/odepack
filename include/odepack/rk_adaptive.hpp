#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method


#include "solver_impl.hpp"

template<typename T, int Norder, int Nstages> 
using A_matrix = Eigen::Array<T, Nstages, Norder>;

template<typename T, int Norder, int Nstages> 
using B_matrix = Eigen::Array<T, Nstages, 1>;

template<typename T, int Norder, int Nstages> 
using C_matrix = B_matrix<T, Norder, Nstages>;

template<typename T, int Norder, int Nstages> 
using E_matrix = Eigen::Array<T, Nstages+1, 1>;

template<typename T, int Nstages> 
using P_matrix = Eigen::Array<T, Nstages+1, -1>;



template<typename DerivedRK, typename T, int N, int Nstages, int Norder>
class RungeKutta;


template<typename T, int N, int Nstages, int Norder>
class RKState final: public DerivedState<T, N, RKState<T, N, Nstages, Norder>>{

    using StageContainer = std::array<vec<T, N>, Nstages+1>;

    template<typename, typename, int, int, int>
    friend class RungeKutta;

public:

    RKState(const T& t, const vec<T, N>& q, const T& h);

private:

    StageContainer _K;

};


template<typename T, int N, int Nstages, int Norder>
struct _RK_mutable{

    _RK_mutable(const vec<T, N>& q);

    vec<T, N> qabs;
    vec<T, N> scale;
    vec<T, N> dq;
    vec<T, N> err_sum;
    vec<T, N> err;
    RKState<T, N, Nstages, Norder> state;

};


template<typename RKDerived, typename T, int N, int Nstages, int Norder>
class RungeKutta : public DerivedSolver<T, N, RKDerived, RKState<T, N, Nstages, Norder>, StandardLocalInterpolator<T, N, RKState<T, N, Nstages, Norder>>>{
    using STATE = RKState<T, N, Nstages, Norder>;

public:

    using INTERPOLATOR = StandardLocalInterpolator<T, N, RKState<T, N, Nstages, Norder>>;

    static constexpr int ERR_EST_ORDER = RKDerived::ERR_EST_ORDER;
    static const T ERR_EXP;

    using OdsBase = DerivedSolver<T, N, RKDerived, STATE, INTERPOLATOR>;
    using StageContainer = std::array<vec<T, N>, Nstages+1>;
    using Atype = A_matrix<T, Norder, Nstages>;
    using Btype = B_matrix<T, Norder, Nstages>;
    using Ctype = C_matrix<T, Norder, Nstages>;
    using Etype = E_matrix<T, Norder, Nstages>;
    using Ptype = P_matrix<T, Nstages>;

    static const Atype A;
    static const Btype B;
    static const Ctype C;
    static const Etype E;
    static const Ptype P;

    void adapt_impl(STATE& res, const STATE& state);

    STATE new_state(const T& t, const vec<T, N>& q, const T& h) const;

    void coef_matrix(Eigen::Matrix<T, N, -1>& mat, const STATE& state1, const STATE& state2) const;

protected:

    RungeKutta(SOLVER_CONSTRUCTOR(T, N)) : OdsBase(name, ARGS), _rk_mut(q0) {}

    DEFAULT_RULE_OF_FOUR(RungeKutta);

private:

    void _step_impl(STATE& result, const STATE& state, const T& h) const;

    mutable _RK_mutable<T, N, Nstages, Norder> _rk_mut;

};



template<typename T, int N>
class RK45 : public RungeKutta<RK45<T, N>, T, N, 6, 5>{

    static const int Norder = 5;
    static const int Nstages = 6;
    

    using RKbase = RungeKutta<RK45<T, N>, T, N, Nstages, Norder>;
    using INTERPOLATOR = RKbase::INTERPOLATOR;
    using STATE = RKState<T, N, Nstages, Norder>;
    

public:

    static constexpr int ERR_EST_ORDER = 4;
    static constexpr int INTERP_ORDER = 4;

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(RK45);
    
    static typename RKbase::Atype Amatrix();

    static typename RKbase::Btype Bmatrix();

    static typename RKbase::Ctype Cmatrix();

    static typename RKbase::Etype Ematrix();

    static typename RKbase::Ptype Pmatrix();

};




template<typename T, int N>
class RK23 : public RungeKutta<RK23<T, N>, T, N, 3, 3> {

    static const int Norder = 3;
    static const int Nstages = 3;
    

    using RKbase = RungeKutta<RK23<T, N>, T, N, Nstages, Norder>;
    using INTERPOLATOR = RKbase::INTERPOLATOR;
    using STATE = RKState<T, N, Nstages, Norder>;
    
public:

    static constexpr int ERR_EST_ORDER = 2;
    static constexpr int INTERP_ORDER = 3;

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(RK23);

    static typename RKbase::Atype Amatrix();

    static typename RKbase::Btype Bmatrix();

    static typename RKbase::Ctype Cmatrix();

    static typename RKbase::Etype Ematrix();

    static typename RKbase::Ptype Pmatrix();

};











//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//--------------------------------IMPLEMENTATIONS-------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------












template<typename T, int N, int Nstages, int Norder>
RKState<T, N, Nstages, Norder>::RKState(const T& t, const vec<T, N>& q, const T& h) : DerivedState<T, N, RKState>(t, q, h){
    for (int i=0; i<Nstages+1; i++){
        _K[i].resize(q.size());
    }
}


template<typename T, int N, int Nstages, int Norder>
_RK_mutable<T, N, Nstages, Norder>::_RK_mutable(const vec<T, N>& q):qabs(q), scale(q), dq(q), err_sum(q), err(q), state(0, q, 0){}


template<typename RKDerived, typename T, int N, int Nstages, int Norder>
void RungeKutta<RKDerived, T, N, Nstages, Norder>::adapt_impl(STATE& res, const STATE& state){
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const T& atol = this->atol();
    const T& rtol = this->rtol();
    const vec<T, N>& q = state.vector();

    res._direction = state.direction();
    _rk_mut.qabs = q.cwiseAbs();
    const vec<T, N>& q_new = res._q;
    T& habs = res._habs;
    habs = state.habs();
    T h, err_norm, factor, _factor;

    bool step_accepted = false;
    bool step_rejected = false;
    while (!step_accepted){

        h = habs * this->direction();

        _step_impl(res, state, h);
        _rk_mut.scale = atol + _rk_mut.qabs.cwiseMax(q_new.cwiseAbs())*rtol;
        _rk_mut.err = res._error / _rk_mut.scale;
        err_norm = rms_norm(_rk_mut.err);
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
        if (!res.resize_step(factor, h_min, max_step)){
            break;
        }
    }
}


template<typename RKDerived, typename T, int N, int Nstages, int Norder>
RungeKutta<RKDerived, T, N, Nstages, Norder>::STATE RungeKutta<RKDerived, T, N, Nstages, Norder>::new_state(const T& t, const vec<T, N>& q, const T& h) const {
    return STATE(t, q, h);
}

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
void RungeKutta<RKDerived, T, N, Nstages, Norder>::coef_matrix(Eigen::Matrix<T, N, -1>& mat, const STATE& state1, const STATE& state2) const{
    for (int i=0; i<N; i++){
        for (int j=0; j<mat.cols(); j++){
            mat(i, j) = 0;
            for (size_t k=0; k<Nstages+1; k++){
                mat(i, j) += state2._K[k][i] * this->P(k, j);
            }
        }
    }
}

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
void RungeKutta<RKDerived, T, N, Nstages, Norder>::_step_impl(STATE& result, const STATE& state, const T& h) const{
    StageContainer& K = result._K;
    this->_rhs(K[0], state.t(), state.vector());
    result._error = K[0] * this->E(0)*h;

    result._q = B(0)*K[0]*h;

    for (size_t s = 1; s < Nstages; s++){
        //calculate df
        _rk_mut.dq = K[0] * this->A(s, 0) * h;
        for (size_t j=1; j<s; j++){
            _rk_mut.dq += this->A(s, j) * K[j] * h;
        }
        //calculate _K
        this->_rhs(K[s], state.t()+C(s)*h, state.vector()+_rk_mut.dq);
        result._error += K[s] * this->E(s)*h;
        result._q += B(s)*K[s]*h;
    }

    result._q += state.vector();
    this->_rhs(K[Nstages], state.t()+h, result._q);
    result._error += K[Nstages] * this->E(Nstages)*h;
    result._t = state.t()+h;
}


template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const T RungeKutta<RKDerived, T, N, Nstages, Norder>::ERR_EXP = T(-1)/T(ERR_EST_ORDER+1);

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const RungeKutta<RKDerived, T, N, Nstages, Norder>::Atype RungeKutta<RKDerived, T, N, Nstages, Norder>::A = RKDerived::Amatrix();

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const RungeKutta<RKDerived, T, N, Nstages, Norder>::Btype RungeKutta<RKDerived, T, N, Nstages, Norder>::B = RKDerived::Bmatrix();

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const RungeKutta<RKDerived, T, N, Nstages, Norder>::Ctype RungeKutta<RKDerived, T, N, Nstages, Norder>::C = RKDerived::Cmatrix();

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const RungeKutta<RKDerived, T, N, Nstages, Norder>::Etype RungeKutta<RKDerived, T, N, Nstages, Norder>::E = RKDerived::Ematrix();

template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const RungeKutta<RKDerived, T, N, Nstages, Norder>::Ptype RungeKutta<RKDerived, T, N, Nstages, Norder>::P = RKDerived::Pmatrix();





template<typename T, int N>
RK45<T, N>::RK45(MAIN_CONSTRUCTOR(T, N)) : RKbase("RK45", ARGS){
    this->_finalize(t0, q0, first_step);
}

template<typename T, int N>
typename RK45<T, N>::RKbase::Atype RK45<T, N>::Amatrix() {
    typename RKbase::Atype A;
    A << T(0),        T(0),        T(0),        T(0),        T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656);
    return A;
}

template<typename T, int N>
typename RK45<T, N>::RKbase::Btype RK45<T, N>::Bmatrix(){
    typename RKbase::Btype B;
    B << T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84);
    return B;
}

template<typename T, int N>
typename RK45<T, N>::RKbase::Ctype RK45<T, N>::Cmatrix(){
    typename RKbase::Ctype C;
    C << T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1);
    return C;
}

template<typename T, int N>
typename RK45<T, N>::RKbase::Etype RK45<T, N>::Ematrix() {
    typename RKbase::Etype E;
    E << T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40);
    return E;
}

template<typename T, int N>
typename RK45<T, N>::RKbase::Ptype RK45<T, N>::Pmatrix() {
    typename RKbase::Ptype P(Nstages+1, 4);
    P <<    T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423);
    return P;
}








template<typename T, int N>
RK23<T, N>::RK23(MAIN_CONSTRUCTOR(T, N)) : RKbase("RK23", ARGS){
    this->_finalize(t0, q0, first_step);
}

template<typename T, int N>
typename RK23<T, N>::RKbase::Atype RK23<T, N>::Amatrix() {
    typename RKbase::Atype A;
    A << T(0),    T(0),    T(0),
            T(1)/T(2), T(0),    T(0),
            T(0),    T(3)/T(4), T(0);
    return A;
}

template<typename T, int N>
typename RK23<T, N>::RKbase::Btype RK23<T, N>::Bmatrix(){
    typename RKbase::Btype B;
    B << T(2)/T(9),
            T(1)/T(3),
            T(4)/T(9);
    return B;
}

template<typename T, int N>
typename RK23<T, N>::RKbase::Ctype RK23<T, N>::Cmatrix(){
    typename RKbase::Ctype C;
    C << T(0),
            T(1)/T(2),
            T(3)/T(4);
    return C;
}

template<typename T, int N>
typename RK23<T, N>::RKbase::Etype RK23<T, N>::Ematrix() {
    typename RKbase::Etype E;
    E << T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8);
    return E;
}

template<typename T, int N>
typename RK23<T, N>::RKbase::Ptype RK23<T, N>::Pmatrix() {
    typename RKbase::Ptype P(Nstages+1, 4);
    P << T(1),   -T(4)/T(3),  T(5)/T(9),
        T(0),    T(1),      -T(2)/T(3),
        T(0),    T(4)/T(3), -T(8)/T(9),
        T(0),   -T(1),       T(1);
    return P;
}

#endif