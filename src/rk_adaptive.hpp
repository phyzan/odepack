#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

#include "odesolvers.hpp"

template<typename Tt, int Norder, int Nstages> 
using A_matrix = Eigen::Array<Tt, Nstages, Norder>;

template<typename Tt, int Norder, int Nstages> 
using B_matrix = Eigen::Array<Tt, Nstages, 1>;

template<typename Tt, int Norder, int Nstages> 
using C_matrix = B_matrix<Tt, Norder, Nstages>;

template<typename Tt, int Norder, int Nstages> 
using E_matrix = Eigen::Array<Tt, Nstages+1, 1>;



template<class Tt, class Ty, int Nstages, int Norder>
class RungeKutta : public OdeSolver<Tt, Ty>{

public:

    using OdsBase = OdeSolver<Tt, Ty>;
    using StageContainer = std::array<Ty, Nstages+1>;
    using Atype = A_matrix<Tt, Norder, Nstages>;
    using Btype = B_matrix<Tt, Norder, Nstages>;
    using Ctype = C_matrix<Tt, Norder, Nstages>;
    using Etype = E_matrix<Tt, Norder, Nstages>;

    virtual Atype Amatrix() const = 0;
    virtual Btype Bmatrix() const = 0;
    virtual Ctype Cmatrix() const = 0;
    virtual Etype Ematrix() const = 0;

    Ty step(const Tt& t_old, const Ty& q_old, const Tt& h) const override{
        return _step(t_old, q_old, h, this->_K);
    }

    State<Tt, Ty> adaptive_step() const override {
        const Tt& t = this->t();
        const Tt& h_min = this->h_min();
        const Tt& h_max = this->h_max();
        const Tt& atol = this->atol();
        const Tt& rtol = this->rtol();
        const Ty& q = this->q_true();
        const Ty qabs = cwise_abs(q);
        Tt habs = this->stepsize();
        Tt h, t_new, err_norm, factor, _factor;
        Ty q_new, scale;

        bool step_accepted = false;
        bool step_rejected = false;
        
        while (!step_accepted){

            h = habs * this->direction();
            t_new = t+h;

            q_new = step(t, q, h);
            scale = atol + cwise_max(qabs, cwise_abs(q_new))*rtol;
            err_norm = _error_norm(_K, h, scale);
            _factor = this->SAFETY*pow(err_norm, err_exp);
            if (err_norm < 1){
                factor = (err_norm == 0) ? this->MAX_FACTOR : std::min(this->MAX_FACTOR, _factor);
                if (step_rejected){
                    factor = factor < 1 ? factor : 1;
                }
                step_accepted = true;
            }
            else {
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

        return {t_new, q_new,  habs};
    }

    const Atype A;
    const Btype B;
    const Ctype C;
    const Etype E;

protected:

    RungeKutta(const SolverArgs<Tt, Ty>& S, const int& err_est_ord, const Atype& A, const Btype& B, const Ctype& C, const Etype& E) : OdsBase(S, Norder, err_est_ord), A(A), B(B), C(C), E(E), err_exp(Tt(-1)/Tt(err_est_ord+1)) {}

    RungeKutta(const RungeKutta<Tt, Ty, Nstages, Norder>& other) : OdsBase(other), A(other.A), B(other.B), C(other.C), E(other.E), _K(other._K), err_exp(other.err_exp){}

    RungeKutta<Tt, Ty, Nstages, Norder> operator=(const RungeKutta<Tt, Ty, Nstages, Norder>& other){
        if (&other == this) return *this;
        _K = other._K;
        OdsBase::operator=(other);
        return *this;
    }

private:

    mutable StageContainer _K;//always holds the value given to it by the last "step" call
    const Tt err_exp;


    Ty _step(const Tt& t_old, const Ty& q_old, const Tt& h, StageContainer& K) const{

        Ty dq;
        K[0] = this->f(t_old, q_old);

        Ty temp = B(0)*K[0];

        for (size_t s = 1; s < Nstages; s++){
            //calculate df
            dq = K[0] * this->A(s, 0) * h;
            for (size_t j=1; j<s; j++){
                dq += this->A(s, j) * K[j] * h;
            }
            //calculate _K
            K[s] = this->f(t_old+this->C(s)*h, q_old+dq);
            temp += B(s)*K[s];
        }

        Ty q_new = q_old + temp*h;
        K[Nstages] = this->f(t_old+h, q_new);
        return q_new;

    };

    Ty _error(const StageContainer& K, const Tt& h) const{
        Ty res = K[0] * this->E(0);
        for (size_t s = 1; s<Nstages+1; s++){
            res += K[s] * this->E(s);
        }
        return res * h;
    }

    Tt _error_norm(const StageContainer& K, const Tt& h, const Ty& scale) const{
        Ty f = _error(K, h) / scale;
        return rms_norm(f);
    }

};



template<class Tt, class Ty>
class RK45 : public RungeKutta<Tt, Ty, 6, 5>{

    static const int Norder = 5;
    static const int Nstages = 6;

    using RKbase = RungeKutta<Tt, Ty, Nstages, Norder>;
    

public:

    RK45(const SolverArgs<Tt, Ty>& S) : RKbase(S, 4, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

    RK45(const RK45<Tt, Ty>& other) : RKbase(other) {}

    RK45<Tt, Ty>& operator=(const RK45<Tt, Ty>& other){
        return RungeKutta<Tt, Ty, Nstages, Norder>::operator=(other);
    }

    RK45<Tt, Ty>* clone() const override{
        return new RK45<Tt, Ty>(*this);
    }
    

    RKbase::Atype Amatrix() const override{
        typename RKbase::Atype A;
        A << Tt(0),        Tt(0),        Tt(0),        Tt(0),        Tt(0),
             Tt(1)/Tt(5),  Tt(0),        Tt(0),        Tt(0),        Tt(0),
             Tt(3)/Tt(40), Tt(9)/Tt(40), Tt(0),        Tt(0),        Tt(0),
             Tt(44)/Tt(45), Tt(-56)/Tt(15), Tt(32)/Tt(9), Tt(0),      Tt(0),
             Tt(19372)/Tt(6561), Tt(-25360)/Tt(2187), Tt(64448)/Tt(6561), Tt(-212)/Tt(729), Tt(0),
             Tt(9017)/Tt(3168), Tt(-355)/Tt(33), Tt(46732)/Tt(5247), Tt(49)/Tt(176), Tt(-5103)/Tt(18656);
        return A;
    }

    RKbase::Btype Bmatrix() const override{
        typename RKbase::Btype B;
        B << Tt(35)/Tt(384),
             Tt(0),
             Tt(500)/Tt(1113),
             Tt(125)/Tt(192),
             Tt(-2187)/Tt(6784),
             Tt(11)/Tt(84);
        return B;
    }

    RKbase::Ctype Cmatrix() const override{
        typename RKbase::Ctype C;
        C << Tt(0),
             Tt(1)/Tt(5),
             Tt(3)/Tt(10),
             Tt(4)/Tt(5),
             Tt(8)/Tt(9),
             Tt(1);
        return C;
    }

    RKbase::Etype Ematrix() const override{
        typename RKbase::Etype E;
        E << Tt(-71)/Tt(57600),
             Tt(0),
             Tt(71)/Tt(16695),
             Tt(-71)/Tt(1920),
             Tt(17253)/Tt(339200),
             Tt(-22)/Tt(525),
             Tt(1)/Tt(40);
        return E;
    }
};




template<class Tt, class Ty>
class RK23 : public RungeKutta<Tt, Ty, 3, 3> {

    static const int Norder = 3;
    static const int Nstages = 3;

    using RKbase = RungeKutta<Tt, Ty, Nstages, Norder>;
    
public:
    RK23(const SolverArgs<Tt, Ty>& S) : RKbase(S, 2, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

    RK23(const RK23<Tt, Ty>& other) : RKbase(other) {}

    RK23<Tt, Ty>& operator=(const RK23<Tt, Ty>& other){
        return RKbase::operator=(other);
    }

    RK23<Tt, Ty>* clone() const override{
        return new RK23<Tt, Ty>(*this);
    }

    RKbase::Atype Amatrix() const override{
        typename RKbase::Atype A;
        A << Tt(0),    Tt(0),    Tt(0),
                Tt(1)/Tt(2), Tt(0),    Tt(0),
                Tt(0),    Tt(3)/Tt(4), Tt(0);
        return A;
    }

    RKbase::Btype Bmatrix() const override{
        typename RKbase::Btype B;
        B << Tt(2)/Tt(9),
                Tt(1)/Tt(3),
                Tt(4)/Tt(9);
        return B;
    }

    RKbase::Ctype Cmatrix() const override{
        typename RKbase::Ctype C;
        C << Tt(0),
                Tt(1)/Tt(2),
                Tt(3)/Tt(4);
        return C;
    }

    RKbase::Etype Ematrix() const override{
        typename RKbase::Etype E;
        E << Tt(5)/Tt(72),
                Tt(-1)/Tt(12),
                Tt(-1)/Tt(9),
                Tt(1)/Tt(8);
        return E;
    }
};



#endif