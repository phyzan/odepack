#include "adaptive_rk.hpp"
#include <iostream>
#include <chrono>


using Tf = vec<double, 1>;

Tf f(const double& t, const Tf& y, const std::vector<double>& args) {
    return Tf{std::cos(t)};
}



int main() {
    // Create an Eigen::Array for the initial condition
    Tf y0 = Tf::Zero();  // initial condition
    // Eigen::Array<double, 1, 1> t_span(0., 5.); // time span [0., 5.]

    // Now pass these Eigen::Array objects as arguments


    RK23<double, 1> solver1(f, y0, {0., 10001*1.57079632679}, 0.1, 0., {}, 1e-5, 1e-10);

    auto start = std::chrono::high_resolution_clock::now();
    while (solver1.is_running()) {
        solver1.advance();
    }
    solver1.state().show();
    
    // After loop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Total runtime: " << elapsed.count() << " seconds\n";

    // std::cout << n << std::endl;
    
}


//g++ -std=c++20 -O3 test.cpp -o test