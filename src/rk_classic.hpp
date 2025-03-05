#ifndef RK_CLASSIC_HPP
#define RK_CLASSIC_HPP

#include "odesolvers.hpp"


template<class Tt, class Ty>
class RKClassic : public OdeSolver<Tt, Ty>{

public:

    using OdsBase = OdeSolver<Tt, Ty>;

    const int lte;
    const Tt err_exp;

    State<Tt, Ty> adaptive_step() const override {
        Tt habs = this->stepsize;
        Tt h, t_new, factor, _factor, err_norm;
        Ty q_new, q_half, q_new_double, accepted_err;
        bool step_accepted = false;
        bool step_rejected = false;

        while (!step_accepted){

            h = habs * this->direction;
            t_new = this->t+h;

            q_new = this->step(this->t, this->q, h);
            q_half = this->step(this->t, this->q, h/2);
            q_new_double = this->step(this->t+h/2, q_half, h/2);

            accepted_err = this->atol + (q_new - q_new_double).cwiseAbs()*this->rtol;
            err_norm = ((q_new - q_new_double)/accepted_err).cwiseAbs().maxCoeff();
            _factor = this->SAFETY*std::pow(err_norm, err_exp);
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
        }
        return {t_new, q_new,  habs};
    }

protected:

    RKClassic(const SolverArgs<Tt, Ty>& S, const int& lte) : OdsBase(S), lte(lte), err_exp(Tt(-1)/lte) {}

};


template<class Tt, class Ty>
class RK4 : public RKClassic<Tt, Ty>{
    
    using RKbase = RKClassic<Tt, Ty>;

public:

    RK4(const SolverArgs<Tt, Ty>& S) : RKbase(S, 5) {}

    Ty step(const Tt& t_old, const Ty& q_old, const Tt& h) const override{
        Ty k1 = this->f(t_old, q_old, this->args);
        Ty k2 = this->f(t_old+h/2, q_old+h*k1/2, this->args);
        Ty k3 = this->f(t_old+h/2, q_old+h*k2/2, this->args);
        Ty k4 = this->f(t_old+h, q_old+h*k3, this->args);
        return q_old + h/6*(k1+2*k2+2*k3+k4);
    }

};



#endif