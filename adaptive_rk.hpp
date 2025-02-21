#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP


#include "odesolvers.hpp"
#include <iostream>

template<typename Tt, int Norder, int Nstages> 
using A_matrix = Eigen::Array<Tt, Nstages, Norder>;

template<typename Tt, int Norder, int Nstages> 
using B_matrix = Eigen::Array<Tt, Nstages, 1>;

template<typename Tt, int Norder, int Nstages> 
using C_matrix = B_matrix<Tt, Norder, Nstages>;

template<typename Tt, int Norder, int Nstages> 
using E_matrix = Eigen::Array<Tt, Nstages+1, 1>;




template<class RK, class Tt, int Nstages, int Norder, size_t N = 0, typename Callable = ode<Tt, N>>
class RungeKutta : public OdeSolver<RungeKutta<RK, Tt, Nstages, Norder, N>, Tt, N, Callable>{              

public:

    using OdsBase = OdeSolver<RungeKutta<RK, Tt, Nstages, Norder, N>, Tt, N, Callable>;
    using typename OdsBase::Ty;
    using StageContainer = std::array<Ty, Nstages+1>;
    using Atype = A_matrix<Tt, Norder, Nstages>;
    using Btype = B_matrix<Tt, Norder, Nstages>;
    using Ctype = C_matrix<Tt, Norder, Nstages>;
    using Etype = E_matrix<Tt, Norder, Nstages>;


    const Tt rtol;
    const Tt atol;
    const Atype A = RK::Amatrix();
    const Btype B = RK::Bmatrix();
    const Ctype C = RK::Cmatrix();
    const Etype E = RK::Ematrix();


    Ty step(const Tt& t_old, const Ty& y_old, const Tt& h){
        return _step(t_old, y_old, h, this->_K);
    }

    void advance(){
        const Tt t = this->t_now();
        const Ty y = this->y_now();
        Tt habs = this->h_now()*this->direction;
        const Ty yabs = y.cwiseAbs();
        Tt h;
        Tt t_new;
        Ty y_new;
        Tt err_norm;
        Tt scale;
        Tt factor;

        bool step_accepted = false;
        bool step_rejected = false;

        while (!step_accepted){

            h = habs * this->direction;
            t_new = t+h;

            y_new = step(t, y, h);
            scale = atol + std::max(yabs.maxCoeff(), y_new.cwiseAbs().maxCoeff())*rtol;
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

        this->_update(t_new, y_new, habs*this->direction);
    }


protected:

    RungeKutta(Callable&& func, const Ty& y0, const Tt (&span)[2], const Tt& h, const Tt& min_h, const std::vector<Tt>& args, const Tt& rtol, const Tt& atol) : OdsBase(std::forward<Callable>(func), y0, span, h, min_h, args), rtol(rtol), atol(atol) {}

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

    Tt _error_norm(const StageContainer& K, const Tt& h, const Tt& scale) const{
        Ty f = _error(K, h) / scale;
        return std::sqrt((f * f).sum() / f.size());
    }

public:

    static Atype Amatrix() {return RK::Amatrix();}
    static Btype Bmatrix() {return RK::Bmatrix();}
    static Ctype Cmatrix() {return RK::Cmatrix();}
    static Etype Ematrix() {return RK::Ematrix();}
};



template<class Tt, size_t N = 0, typename Callable = ode<Tt, N>>
class RK45 : public RungeKutta<RK45<Tt, N, Callable>, Tt, 6, 5, N, Callable>{

    static const int Norder = 5;
    static const int Nstages = 6;

    using RKbase = RungeKutta<RK45<Tt, N, Callable>, Tt, Nstages, Norder, N, Callable>;
    

public:
    RK45(Callable func, const typename RKbase::Ty& y0, const Tt (&span)[2], const Tt& h, const Tt& min_h, const std::vector<Tt>& args, const Tt& rtol, const Tt& atol) : RungeKutta<RK45<Tt, N, Callable>, Tt, Nstages, Norder, N, Callable>(std::forward<Callable>(func), y0, span, h, min_h, args, rtol, atol){}

    static RKbase::Atype Amatrix() {
        typename RKbase::Atype A;
        A << Tt(0),        Tt(0),        Tt(0),        Tt(0),        Tt(0),
             Tt(1)/Tt(5),  Tt(0),        Tt(0),        Tt(0),        Tt(0),
             Tt(3)/Tt(40), Tt(9)/Tt(40), Tt(0),        Tt(0),        Tt(0),
             Tt(44)/Tt(45), Tt(-56)/Tt(15), Tt(32)/Tt(9), Tt(0),      Tt(0),
             Tt(19372)/Tt(6561), Tt(-25360)/Tt(2187), Tt(64448)/Tt(6561), Tt(-212)/Tt(729), Tt(0),
             Tt(9017)/Tt(3168), Tt(-355)/Tt(33), Tt(46732)/Tt(5247), Tt(49)/Tt(176), Tt(-5103)/Tt(18656);
        return A;
    }

    static RKbase::Btype Bmatrix() {
        typename RKbase::Btype B;
        B << Tt(35)/Tt(384),
             Tt(0),
             Tt(500)/Tt(1113),
             Tt(125)/Tt(192),
             Tt(-2187)/Tt(6784),
             Tt(11)/Tt(84);
        return B;
    }

    static RKbase::Ctype Cmatrix() {
        typename RKbase::Ctype C;
        C << Tt(0),
             Tt(1)/Tt(5),
             Tt(3)/Tt(10),
             Tt(4)/Tt(5),
             Tt(8)/Tt(9),
             Tt(1);
        return C;
    }

    static RKbase::Etype Ematrix() {
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




template<class Tt, size_t N = 0, typename Callable = ode<Tt, N>>
class RK23 : public RungeKutta<RK23<Tt, N, Callable>, Tt, 3, 3, N, Callable> {

    static const int Norder = 3;
    static const int Nstages = 3;

    using RKbase = RungeKutta<RK23<Tt, N, Callable>, Tt, Nstages, Norder, N, Callable>;
    
public:
    RK23(Callable func, const typename RKbase::Ty& y0, const Tt (&span)[2], const Tt& h, const Tt& min_h, const std::vector<Tt>& args, const Tt& rtol, const Tt& atol) 
        : RungeKutta<RK23<Tt, N, Callable>, Tt, Nstages, Norder, N, Callable>(std::forward<Callable>(func), y0, span, h, min_h, args, rtol, atol) {}

    static RKbase::Atype Amatrix() {
        typename RKbase::Atype A;
        A << Tt(0),    Tt(0),    Tt(0),
                Tt(1)/Tt(2), Tt(0),    Tt(0),
                Tt(0),    Tt(3)/Tt(4), Tt(0);
        return A;
    }

    static RKbase::Btype Bmatrix() {
        typename RKbase::Btype B;
        B << Tt(2)/Tt(9),
                Tt(1)/Tt(3),
                Tt(4)/Tt(9);
        return B;
    }

    static RKbase::Ctype Cmatrix() {
        typename RKbase::Ctype C;
        C << Tt(0),
                Tt(1)/Tt(2),
                Tt(3)/Tt(4);
        return C;
    }

    static RKbase::Etype Ematrix() {
        typename RKbase::Etype E;
        E << Tt(5)/Tt(72),
                Tt(-1)/Tt(12),
                Tt(-1)/Tt(9),
                Tt(1)/Tt(8);
        return E;
    }
};



#endif