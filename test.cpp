#include "src/ode.hpp"

using Tt = double;
using Tf = vec<Tt, 4>;

Tf f(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return {q[2], q[3], -q[0], -q[1]};
    // return q;
}

Tt fevent(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return f[1];
}

bool check_fevent(const Tt& t, const Tf& f, const std::vector<Tt>& args){
    return f[3]>0;
}


int main(){

    double pi = 3.14159265359;
    double t_max = 10001*pi/2;
    Tf q0 = {1, 1, 2.3, 4.5};
    
    Event<Tt, Tf> event1("Event1", fevent, nullptr, 1, 0);


    // StopEvent<Tt, Tf> event2("stopevent", fevent);

    ODE<Tt, Tf> ode(f, 0, q0, 1e-2, 1e-6, 1e-12, 1e-8, {}, "RK45", 1e-10, {event1});
    ODE<Tt, Tf> ode2 = ode;


    // ODE<Tt, Tf> ode(f, 0, q0, 1e-2, 1e-5, 1e-10, 0., {}, "RK23", 1e-10, {event1, event2, event3});

    // ode.integrate(t_max, 100).examine();
    // return 0;
    OdeSolver<Tt, Tf>* s = ode.solver();
    s->free();

    std::string _input;
    // std::cout << s->f()(1, {1, 1, 2.3, 4.5}, {});
    // // ode2.free();
    char buffer[100];  // Buffer to store input
     // Read input

    std::string input;  // Convert to std::string

    try {
        double value = std::stod(input);  // Convert to double
        std::cout << "Converted value: " << value << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Invalid input: " << e.what() << '\n';
    }

    // s->set_goal

    while (true){
        s->state().show();
        // std::cin.getline(buffer, 100);
        // input = buffer;
        // double val = std::stod(input);
        s->advance();
        std::cin.get();
    }
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