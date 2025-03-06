#include "src/ode.hpp"

using Tt = double;
using Tf = vec<Tt, 4>;

Tf f(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return {q[2], q[3], -q[0], -q[1]};
    // return q;
}

Tt fevent(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return f[1]-2;
}

Tt fevent2(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return f[1]-3;
}

bool check_fevent(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return f[3]>0;
}

Tf mask(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return {1, 1, 0, 0};
}


int main(){

    double pi = 3.14159265359;
    double t_max = 10001*pi/2;
    Tf q0 = {1, 1, 2.3, 4.5};
    
    Event<Tt, Tf> event1("Event1", fevent, nullptr, mask);
    StopEvent<Tt, Tf> event2("Event2", fevent);

    ODE<Tt, Tf> ode(f, 0, q0, 1e-2, 1e-6, 1e-12, 1e-8, {}, "RK45", 1e-10, {event1});
    ODE<Tt, Tf> ode2 = ode;

    ode.integrate(100).examine();
    ode2.integrate(10).examine();

    ode.state().show();
    ode2.state().show();
}