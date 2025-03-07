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
    Event<Tt, Tf> event2("Event2", {1-1e-13, 2-1e-10, 3-1e-10}, nullptr, mask);
    Event<Tt, Tf> event3("Event3", {1, 2, 3}, nullptr, mask);


    // StopEvent<Tt, Tf> event2("stopevent", fevent);

    ODE<Tt, Tf> ode(f, 0, q0, 1e-2, 1e-6, 1e-12, 1e-8, {}, "RK45", 1e-10, {event1, event2, event3});
    ODE<Tt, Tf> ode2 = ode;


    // ODE<Tt, Tf> ode(f, 0, q0, 1e-2, 1e-5, 1e-10, 0., {}, "RK23", 1e-10, {event1, event2, event3});

    // ode.integrate(t_max, 100).examine();
    // OdeSolver<Tt, Tf>* s = ode.solver();
    // s->free();
    // std::cout << s->f()(1, {1, 1, 2.3, 4.5}, {});
    // // ode2.free();
    // while (true){
    //     s->state().show();
    //     s->advance();
    //     std::cin.get();
    // }
    // ode.integrate(10).examine();
    // ode.state().show();
    // ode.free();
    // while (true){
    //     ode2.state().show();
    //     ode2.advance();
    //     std::cin.get();
    // }

    // ode.integrate(10).examine();
    // ode.integrate(8.56);
    // ode.integrate(8.56);
    // ode.state().show();
    ode.state().show();
}