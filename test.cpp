#include "src/ode.hpp"

using Tt = double;
using Ty = vec<Tt, 4>;

Ty f(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return {q[2], q[3], -q[0], -q[1]};
}

Tt ps_func(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[1];
}

Tt stopfunc(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[1]-1.5;
}

bool check(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[3]>0;
}

int main(){
    Tt pi = 3.141592653589793238462;

    Tt t0 = 0;
    Ty q0 = {1., 1., 2.3, 4.5};
    Tt stepsize = 1e-2;
    Tt rtol = 1e-5;
    Tt atol = 1e-10;
    Tt min_step = 0.;

    Tt tmax = 10001*pi/2;

    Event<Tt, Ty> ps("Poincare Section", ps_func, check);
    PeriodicEvent<Tt, Ty> ev2("periodic", 1, 0);
    StopEvent<Tt, Ty> stopev("stop", stopfunc);
    PeriodicEvent<Tt, Ty> ev3("periodic2", 1, 0.001);

    ODE<Tt, Ty> ode(f, t0, q0, stepsize, rtol, atol, min_step, {}, "RK23", 1e-10);
    ode.integrate(tmax, 100).examine();

    // ode.free();
    // while (true){
    //     ode.state().show();
    //     std::cin.get();
    //     ode.free();
    //     ode.advance();
    // }

    // ode.examine();

}