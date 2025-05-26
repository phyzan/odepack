#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

#include "odesolvers.hpp"

template<typename T, int Norder, int Nstages> 
using A_matrix = Eigen::Array<T, Nstages, Norder>;

template<typename T, int Norder, int Nstages> 
using B_matrix = Eigen::Array<T, Nstages, 1>;

template<typename T, int Norder, int Nstages> 
using C_matrix = B_matrix<T, Norder, Nstages>;

template<typename T, int Norder, int Nstages> 
using E_matrix = Eigen::Array<T, Nstages+1, 1>;



template<class T, int N, int Nstages, int Norder>
class RungeKutta : public OdeSolver<T, N>{

public:

    using OdsBase = OdeSolver<T, N>;
    using StageContainer = std::array<vec<T, N>, Nstages+1>;
    using Atype = A_matrix<T, Norder, Nstages>;
    using Btype = B_matrix<T, Norder, Nstages>;
    using Ctype = C_matrix<T, Norder, Nstages>;
    using Etype = E_matrix<T, Norder, Nstages>;

    virtual Atype Amatrix() const = 0;
    virtual Btype Bmatrix() const = 0;
    virtual Ctype Cmatrix() const = 0;
    virtual Etype Ematrix() const = 0;


    inline void step_impl(vec<T, N>& result, const T& t_old, const vec<T, N>& q_old, const T& h) const override{
        this->_step_impl(result, t_old, q_old, h, _K);
    }

    void adapt_impl(State<T, N>& result, const State<T, N>& state) const override {
        const T& t = this->t();
        const T& h_min = this->h_min();
        const T& h_max = this->h_max();
        const T& atol = this->atol();
        const T& rtol = this->rtol();
        const vec<T, N>& q = state.q;
        _qabs = q.cwiseAbs();
        _scale.resize(q.size());
        T& t_new = result.t;
        vec<T, N>& q_new = result.q;
        T& habs = result.h_next;
        habs = state.h_next;
        T h, err_norm, factor, _factor;

        bool step_accepted = false;
        bool step_rejected = false;
        while (!step_accepted){

            h = habs * this->direction();
            t_new = t+h;

            step_impl(q_new, t, q, h);
            _scale = atol + _qabs.cwiseMax(q_new.cwiseAbs())*rtol;
            err_norm = _error_norm(_K, h, _scale);
            _factor = this->SAFETY*pow(err_norm, err_exp);
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
            habs *= factor;
            if (habs > h_max){
                habs = h_max;
            }
            if (habs < h_min){
                habs = h_min;
                break;
            }
            
        }
    }

    const Atype A;
    const Btype B;
    const Ctype C;
    const Etype E;

protected:

    RungeKutta(const SolverArgs<T, N>& S, const int& err_est_ord, const Atype& A, const Btype& B, const Ctype& C, const Etype& E) : OdsBase(S, Norder, err_est_ord), A(A), B(B), C(C), E(E), err_exp(T(-1)/T(err_est_ord+1)) {}

    RungeKutta(const RungeKutta<T, N, Nstages, Norder>& other) : OdsBase(other), A(other.A), B(other.B), C(other.C), E(other.E), _K(other._K), err_exp(other.err_exp){}

    RungeKutta<T, N, Nstages, Norder> operator=(const RungeKutta<T, N, Nstages, Norder>& other){
        if (&other == this) return *this;
        _K = other._K;
        OdsBase::operator=(other);
        return *this;
    }

private:

    void _step_impl(vec<T, N>& q_new, const T& t_old, const vec<T, N>& q_old, const T& h, StageContainer& K) const{

        this->_apply_jac(K[0], t_old, q_old);

        q_new = B(0)*K[0]*h;

        for (size_t s = 1; s < Nstages; s++){
            //calculate df
            _dq = K[0] * this->A(s, 0) * h;
            for (size_t j=1; j<s; j++){
                _dq += this->A(s, j) * K[j] * h;
            }
            //calculate _K
            this->_apply_jac(K[s], t_old+this->C(s)*h, q_old+_dq);
            // K[s] = this->_jac(t_old+this->C(s)*h, q_old+_dq);
            q_new += B(s)*K[s]*h;
        }

        q_new += q_old;
        this->_apply_jac(K[Nstages], t_old+h, q_new);
        // K[Nstages] = this->_jac(t_old+h, q_new);
    }

    T _error_norm(const StageContainer& K, const T& h, const vec<T, N>& scale) const{

        //error calc
        _err_sum = K[0] * this->E(0)*h;
        for (size_t s = 1; s<Nstages+1; s++){
            _err_sum += K[s] * this->E(s)*h;
        }

        //error norm calc
        _err = _err_sum / scale;
        return rms_norm(_err);
    }

    mutable StageContainer _K;//always holds the value given to it by the last "step" call
    const T err_exp;

    mutable vec<T, N> _qabs;
    mutable vec<T, N> _scale;
    mutable vec<T, N> _dq;
    mutable vec<T, N> _err_sum;
    mutable vec<T, N> _err;

};



template<class T, int N>
class RK45 : public RungeKutta<T, N, 6, 5>{

    static const int Norder = 5;
    static const int Nstages = 6;

    using RKbase = RungeKutta<T, N, Nstages, Norder>;
    

public:

    RK45(const SolverArgs<T, N>& S) : RKbase(S, 4, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

    RK45(const RK45<T, N>& other) : RKbase(other) {}

    RK45<T, N>& operator=(const RK45<T, N>& other){
        return RungeKutta<T, N, Nstages, Norder>::operator=(other);
    }

    RK45<T, N>* clone() const override{
        return new RK45<T, N>(*this);
    }
    

    RKbase::Atype Amatrix() const override{
        typename RKbase::Atype A;
        A << T(0),        T(0),        T(0),        T(0),        T(0),
             T(1)/T(5),  T(0),        T(0),        T(0),        T(0),
             T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0),
             T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0),
             T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0),
             T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656);
        return A;
    }

    RKbase::Btype Bmatrix() const override{
        typename RKbase::Btype B;
        B << T(35)/T(384),
             T(0),
             T(500)/T(1113),
             T(125)/T(192),
             T(-2187)/T(6784),
             T(11)/T(84);
        return B;
    }

    RKbase::Ctype Cmatrix() const override{
        typename RKbase::Ctype C;
        C << T(0),
             T(1)/T(5),
             T(3)/T(10),
             T(4)/T(5),
             T(8)/T(9),
             T(1);
        return C;
    }

    RKbase::Etype Ematrix() const override{
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




template<class T, int N>
class RK23 : public RungeKutta<T, N, 3, 3> {

    static const int Norder = 3;
    static const int Nstages = 3;

    using RKbase = RungeKutta<T, N, Nstages, Norder>;
    
public:
    RK23(const SolverArgs<T, N>& S) : RKbase(S, 2, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

    RK23(const RK23<T, N>& other) : RKbase(other) {}

    RK23<T, N>& operator=(const RK23<T, N>& other){
        return RKbase::operator=(other);
    }

    RK23<T, N>* clone() const override{
        return new RK23<T, N>(*this);
    }

    RKbase::Atype Amatrix() const override{
        typename RKbase::Atype A;
        A << T(0),    T(0),    T(0),
                T(1)/T(2), T(0),    T(0),
                T(0),    T(3)/T(4), T(0);
        return A;
    }

    RKbase::Btype Bmatrix() const override{
        typename RKbase::Btype B;
        B << T(2)/T(9),
                T(1)/T(3),
                T(4)/T(9);
        return B;
    }

    RKbase::Ctype Cmatrix() const override{
        typename RKbase::Ctype C;
        C << T(0),
                T(1)/T(2),
                T(3)/T(4);
        return C;
    }

    RKbase::Etype Ematrix() const override{
        typename RKbase::Etype E;
        E << T(5)/T(72),
                T(-1)/T(12),
                T(-1)/T(9),
                T(1)/T(8);
        return E;
    }
};



#endif