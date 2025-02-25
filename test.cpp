#include "ode.hpp"
#include <iostream>
#include <chrono>



using Tf = vec<double, 4>;

Tf f(const double& t, const Tf& y, const std::vector<double>& args) {
    return {y[2], y[3], -y[0], -y[1]};
}

bool getevent(const double& t1, const Tf& f1, const double& t2, const Tf& f2){
    return f1[1] < 0 && f2[1] > 0;
}


int main() {    
    Tf y0(4);
    y0 << 1, 1, 2.3, 4.5;
    double pi = 3.14159265359;
    double t_max = 10001*pi/2;

    ODE<double, Tf, true, true> ode(f);

    ICS<double, Tf> ics = {0., y0};
    OdeArgs<double, Tf, true> args = {ics, t_max, 0.001, 1e-5, 1e-10, 0., "RK23", 400, {}};


    OdeResult<double, Tf> res = ode.solve(args);

    std::cout << std::endl << res.runtime << "\n";
    std::cout << res.y.size() << "\n\n";
    // std::cout << res.y[res.y.size()-1];
    
}


// g++ -std=c++20 -O3 -fopenmp -Wall -fPIC test.cpp -o test good

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) test.cpp -o pytest$(python3-config --extension-suffix)