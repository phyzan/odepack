#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP


#include "odesolvers.hpp"

template<typename Tt, int Norder, int Nstages> 
using A_matrix = Eigen::Array<Tt, Nstages, Norder>;

template<typename Tt, int Norder, int Nstages> 
using B_matrix = Eigen::Array<Tt, Nstages, 1>;

template<typename Tt, int Norder, int Nstages> 
using C_matrix = B_matrix<Tt, Norder, Nstages>;

template<typename Tt, int Norder, int Nstages> 
using E_matrix = Eigen::Array<Tt, Nstages+1, 1>;



template<class Tt, int Nstages, int Norder, size_t N, bool raw_ode, bool raw_event>
class RungeKutta : public OdeSolver<Tt, N, raw_ode, raw_event>{

public:

    using OdsBase = OdeSolver<Tt, N, raw_ode, raw_event>;
    using typename OdsBase::Ty;
    using StageContainer = std::array<Ty, Nstages+1>;
    using Atype = A_matrix<Tt, Norder, Nstages>;
    using Btype = B_matrix<Tt, Norder, Nstages>;
    using Ctype = C_matrix<Tt, Norder, Nstages>;
    using Etype = E_matrix<Tt, Norder, Nstages>;

    virtual Atype Amatrix() const = 0;
    virtual Btype Bmatrix() const = 0;
    virtual Ctype Cmatrix() const = 0;
    virtual Etype Ematrix() const = 0;

    Ty step(const Tt& t_old, const Ty& y_old, const Tt& h) const override{
        return _step(t_old, y_old, h, this->_K);
    }

    State<Tt, N> adaptive_step() const override {
        const Tt t = this->t_now();
        const Ty y = this->y_now();
        Tt habs = this->h_now()*this->direction;
        const Ty yabs = y.cwiseAbs();
        Tt h;
        Tt t_new;
        Ty y_new;
        Tt err_norm;
        Ty scale;
        Tt factor;

        bool step_accepted = false;
        bool step_rejected = false;
        
        while (!step_accepted){

            h = habs * this->direction;
            t_new = t+h;

            y_new = step(t, y, h);
            scale = this->atol + yabs.cwiseMax(y_new.cwiseAbs())*this->rtol;
            err_norm = _error_norm(_K, h, scale);
            if (err_norm < 1){
                if (err_norm == 0){
                    factor = OdsBase::MAX_FACTOR;
                }
                else{
                    factor = std::min(OdsBase::MAX_FACTOR, OdsBase::SAFETY*std::pow(err_norm, err_exp));
                }

                if (step_rejected){
                    factor = factor < 1 ? factor : 1;
                }
                habs *= factor;
                step_accepted = true;
            }
            else {
                habs *= std::max(OdsBase::MIN_FACTOR, OdsBase::SAFETY*std::pow(err_norm, err_exp));
                step_rejected = true;
            }
        }

        return {t_new, y_new,  habs*this->direction};
    }

    const Atype A;
    const Btype B;
    const Ctype C;
    const Etype E;

protected:

    RungeKutta(const SolverArgs<Tt, N, raw_ode, raw_event>& S, const Atype& A, const Btype& B, const Ctype& C, const Etype& E) : OdsBase(S), A(A), B(B), C(C), E(E) {}

private:

    static const int err_est_ord = Norder-1;
    static constexpr Tt err_exp = Tt(-1)/Norder;
    mutable StageContainer _K;//always holds the value given to it by the last "step" call


    Ty _step(const Tt& t_old, const Ty& y_old, const Tt& h, StageContainer& K) const{

        Ty dy;
        K[0] = this->f(t_old, y_old, this->args);

        Ty temp = B(0)*K[0];

        for (size_t s = 1; s < Nstages; s++){
            //calculate df
            dy = K[0] * this->A(s, 0) * h;
            for (size_t j=1; j<s; j++){
                dy += this->A(s, j) * K[j] * h;
            }
            //calculate _K
            K[s] = this->f(t_old+this->C(s)*h, y_old+dy, this->args);
            temp += B(s)*K[s];
        }

        Ty y_new = y_old + temp*h;


        K[Nstages] = this->f(t_old+h, y_new, this->args);

        return y_new;

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
        return std::sqrt((f * f).sum() / f.size());
    }

};



template<class Tt, size_t N = 0, bool raw_ode = true, bool raw_event = true>
class RK45 : public RungeKutta<Tt, 6, 5, N, raw_ode, raw_event>{

    static const int Norder = 5;
    static const int Nstages = 6;

    using RKbase = RungeKutta<Tt, Nstages, Norder, N, raw_ode, raw_event>;
    

public:
    RK45(const SolverArgs<Tt, N, raw_ode, raw_event>& S) : RungeKutta<Tt, Nstages, Norder, N, raw_ode, raw_event>(S, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

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




template<class Tt, size_t N = 0, bool raw_ode = true, bool raw_event = true>
class RK23 : public RungeKutta<Tt, 3, 3, N, raw_ode, raw_event> {

    static const int Norder = 3;
    static const int Nstages = 3;

    using RKbase = RungeKutta<Tt, Nstages, Norder, N, raw_ode, raw_event>;
    
public:
    RK23(const SolverArgs<Tt, N, raw_ode, raw_event>& S) : RungeKutta<Tt, Nstages, Norder, N, raw_ode, raw_event>(S, Amatrix(), Bmatrix(), Cmatrix(), Ematrix()){}

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