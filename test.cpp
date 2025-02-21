#include "adaptive_rk.hpp"
#include <iostream>
#include <chrono>


Eigen::Array<double, 1, 1> f(const double& t, const Eigen::Array<double, 1, 1>& y, const std::vector<double>& args) {
    return Eigen::Array<double, 1, 1>{std::cos(t)};
}

int main() {
    // Create an Eigen::Array for the initial condition
    Eigen::Array<double, 1, 1> y0 = Eigen::Array<double, 1, 1>::Zero();  // initial condition
    // Eigen::Array<double, 1, 1> t_span(0., 5.); // time span [0., 5.]

    // Now pass these Eigen::Array objects as arguments
    RK23<double, 1> solver(f, y0, {0., 10001*1.57079632679}, 0.1, 0., {}, 1e-5, 1e-10);
    auto start = std::chrono::high_resolution_clock::now();
    
    while (solver.is_running()) {
        solver.advance();
    }
    solver.state().show();
    
    // After loop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Total runtime: " << elapsed.count() << " seconds\n";
    // std::cout << n << std::endl;
    
}


//g++ -std=c++20 -O3 test.cpp -o test