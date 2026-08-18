#include <odepack/odepack.hpp>

using T = double;
using namespace ode;

// Harmonic oscillator: q0' = q1, q1' = -q0
// Solution from (1, 0): q0 = cos(t), q1 = -sin(t)
static void rhs(T* out, const T& /*t*/, const T* q, const T* /*args*/){
    out[0] = q[1];
    out[1] = -q[0];
}

// ---- Test 1: single objective, find q[0] = 0 --------------------------------
// Expected crossing: cos(t) = 0  =>  t = pi/2 (decreasing, dir=-1)
void test_single_objective(){
    std::cout << "=== test_single_objective ===\n";

    auto obj = [](const T& /*t*/, const T* q, const T* /*args*/) -> T { return q[0]; };
    using ObjFun = decltype(obj);

    ObjFunData<T, ObjFun> obj_data{.func=obj, .dir=-1};  // decreasing crossing

    T y0[2] = {1.0, 0.0};

    ObjectiveSolver<RK45, T, 2, SolverPolicy::Static, OdeData<RhsFunc<T>, std::nullptr_t>, ObjFun> solver(
        {obj_data},
        OdeData{.Rhs = rhs},
        0.0,              // t0
        View1D<T, 2>{y0}, // q0
        1e-10,            // rtol
        1e-10,            // atol
        0.0,              // min_step
        0.1,              // max_step
        0.0,              // stepsize (auto)
        1,                // direction (forward)
        std::vector<T>{}  // args
    );

    const T period = 2.0 * M_PI;

    T t_expected = period / 4.0;
    int crossings_found = 0;

    while (solver.is_running() && solver.t() < 5.0 * period){
        solver.advance();
        if (solver.is_at_objective()){
            crossings_found++;
            if (std::abs(solver.t() - t_expected) >= 1e-8){
                std::cerr << "Error: crossing time inaccurate!" << std::endl;
            } else if (std::abs(solver.vector()[0]) >= 1e-8){
                std::cerr << "Error: q[0] not near zero at crossing!" << std::endl;
            } else if (std::abs(solver.vector()[1] + 1.0) >= 1e-6){
                std::cerr << "Error: q[1] not near -1 at crossing!" << std::endl;
            } else {
                std::cout << "  SUCCESSFULLY PASSED CROSSING\n";
            }
            t_expected += period;  // next expected crossing
        }
    }



    if (crossings_found != 5){
        std::cerr << "Error: not all crossings found!" << std::endl;
    } else {
        std::cout << "PASSED: all crossings found\n";
    }
}

// ---- Test 2: two objectives simultaneously ----------------------------------
// From q0(0)=0, q1(0)=1: q0 = sin(t), q1 = cos(t)
// Obj 0: q[0] = 0  at t = pi, 3pi, 5pi, ...        (decreasing, dir=-1); q[1] = -1 there
// Obj 1: q[1] = 0  at t = 3pi/2, 7pi/2, 11pi/2, ... (increasing, dir=+1); q[0] = -1 there
void test_two_objectives(){
    std::cout << "=== test_two_objectives ===\n";

    auto obj0 = [](const T& /*t*/, const T* q, const T* /*args*/) -> T { return q[0]; };  // position
    auto obj1 = [](const T& /*t*/, const T* q, const T* /*args*/) -> T { return q[1]; };  // velocity
    using ObjFun0 = decltype(obj0);
    using ObjFun1 = decltype(obj1);

    ObjFunData<T, ObjFun0> data0{.func=obj0, .dir=-1};  // q[0]=sin(t) decreasing through 0 at t=pi
    ObjFunData<T, ObjFun1> data1{.func=obj1, .dir=1};   // q[1]=cos(t) increasing through 0 at t=3pi/2

    T y0[2] = {0.0, 1.0};

    ObjectiveSolver<RK45, T, 2, SolverPolicy::Static, OdeData<RhsFunc<T>, std::nullptr_t>, ObjFun0, ObjFun1> solver(
        {data0, data1},
        OdeData{.Rhs = rhs},
        0.0,
        View1D<T, 2>{y0}, // q0
        1e-10, 1e-10,  // rtol, atol
        0.0, 0.1, 0.0, // min_step, max_step, stepsize
        1,             // direction
        std::vector<T>{}
    );

    const T period = 2.0 * M_PI;

    T t0_expected = M_PI;              // next expected obj0 (q[0]=0) crossing
    T t1_expected = 3.0 * M_PI / 2.0;  // next expected obj1 (q[1]=0) crossing
    int crossings0_found = 0;
    int crossings1_found = 0;

    while (solver.is_running() && solver.t() < 5.0 * period){
        solver.advance();
        if (solver.is_at_objective()){
            if (solver.current_objective() == 0){
                crossings0_found++;
                if (std::abs(solver.t() - t0_expected) >= 1e-8){
                    std::cerr << "Error: obj0 crossing time inaccurate!" << std::endl;
                } else if (std::abs(solver.vector()[0]) >= 1e-8){
                    std::cerr << "Error: q[0] not near zero at obj0 crossing!" << std::endl;
                } else if (std::abs(solver.vector()[1] + 1.0) >= 1e-6){
                    std::cerr << "Error: q[1] not near -1 at obj0 crossing!" << std::endl;
                } else {
                    std::cout << "  SUCCESSFULLY PASSED OBJ0 CROSSING\n";
                }
                t0_expected += period;
            } else {
                crossings1_found++;
                if (std::abs(solver.t() - t1_expected) >= 1e-8){
                    std::cerr << "Error: obj1 crossing time inaccurate!" << std::endl;
                } else if (std::abs(solver.vector()[1]) >= 1e-8){
                    std::cerr << "Error: q[1] not near zero at obj1 crossing!" << std::endl;
                } else if (std::abs(solver.vector()[0] + 1.0) >= 1e-6){
                    std::cerr << "Error: q[0] not near -1 at obj1 crossing!" << std::endl;
                } else {
                    std::cout << "  SUCCESSFULLY PASSED OBJ1 CROSSING\n";
                }
                t1_expected += period;
            }
        }
    }

    if (crossings0_found != 5 || crossings1_found != 5){
        std::cerr << "Error: not all crossings found!" << std::endl;
    } else {
        std::cout << "PASSED: all crossings found\n";
    }
}

void test_objective_solver(){
    test_single_objective();
    test_two_objectives();
    std::cout << "All tests passed.\n";
}
