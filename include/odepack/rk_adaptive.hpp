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




template<typename DerivedRK, typename T, int N, int Nstages, int Norder>
class RungeKutta;


template<typename T, int N, int Nstages, int Norder>
class RKState final: public DerivedState<T, N, RKState<T, N, Nstages, Norder>>{

    using StageContainer = std::array<vec<T, N>, Nstages+1>;

    template<typename, typename, int, int, int>
    friend class RungeKutta;

public:
    RKState(const T& t, const vec<T, N>& q, const T& h) : DerivedState<T, N, RKState>(t, q, h){
        for (int i=0; i<Nstages+1; i++){
            _K[i].resize(q.size());
        }
    }

private:

    StageContainer _K;

};







template<typename T, int N, int Nstages, int Norder>
struct _RK_mutable{

    _RK_mutable(const vec<T, N>& q):qabs(q), scale(q), dq(q), err_sum(q), err(q), state(0, q, 0) {}

    vec<T, N> qabs;
    vec<T, N> scale;
    vec<T, N> dq;
    vec<T, N> err_sum;
    vec<T, N> err;
    RKState<T, N, Nstages, Norder> state;

};


template<typename RKDerived, typename T, int N, int Nstages, int Norder>
class RungeKutta : public DerivedSolver<T, N, RKDerived, RKState<T, N, Nstages, Norder>>{
    using STATE = RKState<T, N, Nstages, Norder>;

public:

    static constexpr int ERR_EST_ORDER = RKDerived::ERR_EST_ORDER;
    static const T ERR_EXP;

    using OdsBase = DerivedSolver<T, N, RKDerived, STATE>;
    using StageContainer = std::array<vec<T, N>, Nstages+1>;
    using Atype = A_matrix<T, Norder, Nstages>;
    using Btype = B_matrix<T, Norder, Nstages>;
    using Ctype = C_matrix<T, Norder, Nstages>;
    using Etype = E_matrix<T, Norder, Nstages>;

    const Atype A;
    const Btype B;
    const Ctype C;
    const Etype E;

    void adapt_impl(STATE& res, const STATE& state) {
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

    STATE new_state(const T& t, const vec<T, N>& q, const T& h) const {
        return STATE(t, q, h);
    }

    void call_impl(vec<T, N>& res, const T& t, const State<T, N>& state1, const State<T, N>& state2) const{
        //TODO: apply interpolation coefficients, remove static_cast
        _step_impl(_rk_mut.state, static_cast<const STATE&>(state1), t-state1.t());
        res = _rk_mut.state.vector();
    }

protected:

    RungeKutta(SOLVER_CONSTRUCTOR(T, N)) : OdsBase(name, ARGS), A(RKDerived::Amatrix()), B(RKDerived::Bmatrix()), C(RKDerived::Cmatrix()), E(RKDerived::Ematrix()), _rk_mut(q0) {}

private:

    void _step_impl(STATE& result, const STATE& state, const T& h) const{
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

    mutable _RK_mutable<T, N, Nstages, Norder> _rk_mut;

};



template<typename T, int N>
class RK45 : public RungeKutta<RK45<T, N>, T, N, 6, 5>{

    static const int Norder = 5;
    static const int Nstages = 6;
    

    using RKbase = RungeKutta<RK45<T, N>, T, N, Nstages, Norder>;
    

public:

    static constexpr int ERR_EST_ORDER = 4;

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T, N)) : RKbase("RK45", ARGS){}
    
    static typename RKbase::Atype Amatrix() {
        typename RKbase::Atype A;
        A << T(0),        T(0),        T(0),        T(0),        T(0),
             T(1)/T(5),  T(0),        T(0),        T(0),        T(0),
             T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0),
             T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0),
             T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0),
             T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656);
        return A;
    }

    static typename RKbase::Btype Bmatrix(){
        typename RKbase::Btype B;
        B << T(35)/T(384),
             T(0),
             T(500)/T(1113),
             T(125)/T(192),
             T(-2187)/T(6784),
             T(11)/T(84);
        return B;
    }

    static typename RKbase::Ctype Cmatrix(){
        typename RKbase::Ctype C;
        C << T(0),
             T(1)/T(5),
             T(3)/T(10),
             T(4)/T(5),
             T(8)/T(9),
             T(1);
        return C;
    }

    static typename RKbase::Etype Ematrix() {
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
};




template<typename T, int N>
class RK23 : public RungeKutta<RK23<T, N>, T, N, 3, 3> {

    static const int Norder = 3;
    static const int Nstages = 3;
    

    using RKbase = RungeKutta<RK23<T, N>, T, N, Nstages, Norder>;
    
public:

    static constexpr int ERR_EST_ORDER = 2;

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T, N)) : RKbase("RK23", ARGS){}

    static typename RKbase::Atype Amatrix() {
        typename RKbase::Atype A;
        A << T(0),    T(0),    T(0),
                T(1)/T(2), T(0),    T(0),
                T(0),    T(3)/T(4), T(0);
        return A;
    }

    static typename RKbase::Btype Bmatrix(){
        typename RKbase::Btype B;
        B << T(2)/T(9),
                T(1)/T(3),
                T(4)/T(9);
        return B;
    }

    static typename RKbase::Ctype Cmatrix() {
        typename RKbase::Ctype C;
        C << T(0),
                T(1)/T(2),
                T(3)/T(4);
        return C;
    }

    static typename RKbase::Etype Ematrix(){
        typename RKbase::Etype E;
        E << T(5)/T(72),
                T(-1)/T(12),
                T(-1)/T(9),
                T(1)/T(8);
        return E;
    }
};




template<typename RKDerived, typename T, int N, int Nstages, int Norder>
const T RungeKutta<RKDerived, T, N, Nstages, Norder>::ERR_EXP = T(-1)/T(ERR_EST_ORDER+1);

#endif