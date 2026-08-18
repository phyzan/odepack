#include "solvers.hpp"
#include "oscillator_solver.hpp"
#include "objective_solver.hpp"
#include "variational_test.hpp"
#include "lazy_scalar.hpp"

#include <iostream>

int main(){
    test_solvers();
    std::cout << " Test 1 completed\n" << std::endl;

    test_oscillator_solver();
    std::cout << " Test 2 completed\n" << std::endl;

    test_objective_solver();
    std::cout << " Test 3 completed\n" << std::endl;

    test_variational_solver();
    std::cout << " Test 4 completed\n" << std::endl;

    test_lazy_scalar();
    std::cout << " Test 5 completed\n" << std::endl;
    return 0;
}
