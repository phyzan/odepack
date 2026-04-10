#include "../include/odepack.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using T = double;
using namespace ode;

// Harmonic oscillator: q0' = q1, q1' = -q0
// Solution from (1, 0): q0 = cos(t), q1 = -sin(t)
void rhs(T* out, const T& t, const T* q, const T* args){
    out[0] = q[1];
    out[1] = -q[0];
}

// ---- Test 1: single objective, find q[0] = 0 --------------------------------
// Expected crossing: cos(t) = 0  =>  t = pi/2 (decreasing, dir=-1)
void test_single_objective(){
    std::cout << "=== test_single_objective ===\n";

    auto obj = [](const T& t, const T* q, const T*) -> T { return q[0]; };
    using ObjFun = decltype(obj);

    ObjFunData<T, ObjFun> obj_data{.func=obj, .dir=-1};  // decreasing crossing

    T y0[2] = {1.0, 0.0};

    ObjectiveSolver<RK45, T, 2, SolverPolicy::Static, OdeData<RhsFunc<T>, std::nullptr_t>, void, ObjFun> solver(
        {obj_data},
        OdeData{.Rhs = rhs},
        0.0,              // t0
        y0,               // q0
        2,                // nsys
        1e-10,            // rtol
        1e-10,            // atol
        0.0,              // min_step
        0.1,              // max_step
        0.0,              // stepsize (auto)
        1,                // direction (forward)
        std::vector<T>{}  // args
    );

    const T t_expected = M_PI / 2.0;
    bool found = false;

    while (solver.is_running() && solver.t() < t_expected + 1.0){
        solver.advance();
        if (std::abs(solver.vector()[0]) < 1e-8){
            found = true;
            break;
        }
    }

    std::cout << "  Detected at t = " << solver.t() << "  (expected " << t_expected << ")\n";
    std::cout << "  q[0] = " << solver.vector()[0] << "  (expected ~0)\n";
    std::cout << "  q[1] = " << solver.vector()[1] << "  (expected ~-1)\n";

    assert(found && "zero crossing not detected");
    assert(std::abs(solver.t() - t_expected) < 1e-8 && "crossing time inaccurate");
    assert(std::abs(solver.vector()[0])       < 1e-8 && "q[0] not near zero at crossing");
    assert(std::abs(solver.vector()[1] + 1.0) < 1e-6 && "q[1] not near -1 at crossing");

    std::cout << "  PASSED\n\n";
}

// ---- Test 2: two objectives simultaneously ----------------------------------
// Obj 0: q[0] = 0  at t = pi/2  (decreasing, dir=-1)
// Obj 1: q[1] = 0  at t = pi    (increasing, dir=+1, since q[1]=-sin going from -1 back to 0)
void test_two_objectives(){
    std::cout << "=== test_two_objectives ===\n";

    auto obj0 = [](const T& t, const T* q, const T*) -> T { return q[0]; };  // position
    auto obj1 = [](const T& t, const T* q, const T*) -> T { return q[1]; };  // velocity
    using ObjFun0 = decltype(obj0);
    using ObjFun1 = decltype(obj1);

    ObjFunData<T, ObjFun0> data0{.func=obj0, .dir=1};  // q[0]=sin(t) decreasing through 0 at t=pi
    ObjFunData<T, ObjFun1> data1{.func=obj1, .dir=1};  // q[1]=cos(t) decreasing through 0 at t=pi/2

    T y0[2] = {0.0, 1.0};

    ObjectiveSolver<RK45, T, 2, SolverPolicy::Static, OdeData<RhsFunc<T>, std::nullptr_t>, void, ObjFun0, ObjFun1> solver(
        {data0, data1},
        OdeData{.Rhs = rhs},
        0.0, y0, 2,
        1e-10, 1e-10,  // rtol, atol
        0.0, 0.1, 0.0, // min_step, max_step, stepsize
        1,             // direction
        std::vector<T>{}
    );

    // First crossing: q[1]=cos(t)=0 at t=pi/2
    const T t1_expected = M_PI / 2.0;
    bool found1 = false;
    while (solver.advance()){
        if (solver.is_at_objective()){
            print("At objective: t = ", solver.t()/(M_PI), solver.vector()[0], solver.vector()[1], "\n");
            std::cin.get();
        }
    }
    std::cout << "  [obj1] detected at t = " << solver.t() << "  (expected " << t1_expected << ")\n";
    assert(found1 && "obj1 (q[1]=0) not detected");
    assert(std::abs(solver.t() - t1_expected) < 1e-8 && "obj1 crossing time inaccurate");

    // Second crossing: q[0]=sin(t)=0 at t=pi
    const T t2_expected = M_PI;
    bool found2 = false;
    while (solver.is_running() && solver.t() < t2_expected + 0.5){
        solver.advance();
        if (std::abs(solver.vector()[0]) < 1e-8){
            found2 = true;
            break;
        }
    }
    std::cout << "  [obj0] detected at t = " << solver.t() << "  (expected " << t2_expected << ")\n";
    std::cout << "  q[0] = " << solver.vector()[0] << "  (expected ~0)\n";
    std::cout << "  q[1] = " << solver.vector()[1] << "  (expected ~-1)\n";
    assert(found2 && "obj0 (q[0]=0) not detected");
    assert(std::abs(solver.t() - t2_expected) < 1e-8 && "obj0 crossing time inaccurate");
    assert(std::abs(solver.vector()[1] + 1.0) < 1e-6 && "q[1] not near -1 at t=pi");

    std::cout << "  PASSED\n\n";
}

int main(){
    test_single_objective();
    test_two_objectives();
    std::cout << "All tests passed.\n";
    return 0;
}
