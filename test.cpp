#include "ode.hpp"
#include <iostream>
#include <chrono>


const size_t n = 4;

using Tf = vec<double, n>;

Tf f(const double& t, const Tf& y, const std::vector<double>& args) {
    return {y[2], y[3], -y[0], -y[1]};
}


int main() {    
    Tf y0(n);
    y0 << 1, 1, 2.3, 4.5;
    double pi = 3.14159265359;
    double t_max = 10001*pi/2;

    ODE<double, n> ode(f);

    ICS<double, n> ics = {0., y0};
    OdeArgs<double, n> args = {ics, t_max, 0.1, 0., 1e-10, 0., "RK45"};

    OdeResult<double, n> res = ode.solve(args);

    std::cout << std::endl << res.runtime << "\n";
    std::cout << res.y.size() << "\n\n";
    std::cout << res.y[res.y.size()-1];
    
}


// g++ -std=c++20 -O3 -Wall -fPIC  -fopenmp test.cpp -o test bad
// g++ -std=c++20 -O3 -Wall -fPIC test.cpp -o test good

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) test.cpp -o pytest$(python3-config --extension-suffix)