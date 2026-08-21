#include <odepack/odepack.hpp>
#include <lazy/apps/mpfrLazy.hpp>

using namespace ode;

using pbox::Box;


template<typename T>
void run(const char* label){

    std::array<T, 2> q0 = {10, 0};
    
    // Let's create a solver for the simple harmonic oscillator using the RK45 method
    Box<OdeSolver<T, 2>> solver = make_vsolver(
        Integrator::RK45,
        OdeData{
            .Rhs=[](auto* dq_dt, const auto& t, const auto* q){
                dq_dt[0] = q[1];
                dq_dt[1] = -q[0];
            },
        },
        T{0},
        View1D<T, 2>{q0.data()},
        T{1e-10},
        T{1e-10}
    );

    // Compute the total runtime for integrating up to t=1000
    auto t0 = std::chrono::high_resolution_clock::now();
    solver->do_advance_until(1000);
    auto t1 = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n\n========== Using " << "\033[1m" << label << "\033[0m" << " precision ==========\n" <<
    "Total runtime: " << "\033[1m" << runtime << " ms\033[0m\n";
    std::cout << "Final state: ";
    solver->show_state(10);
    std::cout << "\n";
}


int main(){

    // Run the solver with different precisions

    // Run with double precision (most common case)
    run<double>("double");

    // Run with long double precision
    run<long double>("long double");

    // Run with arbitrary precision using mpfr::mpreal (set precision to 100 bits)
    mpfr::mpreal::set_default_prec(100);
    run<mpfr::mpreal>("mpfr::mpreal");

    /*
    Run with arbitrary precision using lazy::LazyType<mpfr::mpreal> (set precision to 100 bits)
    This is a lazy evaluation wrapper, and drastically reduces the intermediate termporary
    allocations needed to evaluate an algebraic expressions, which become apparent
    when the numerical type performs heap allocations (like mpfr::mpreal does).

    The lazy::LazyType<mpfr::mpreal> type is a drop-in replacement for mpfr::mpreal, and can be used
    in almost any context where mpfr::mpreal is used, producing the exact same results, but with significantly improved performance in many cases.

    Calling lazy::set_default_mpreal_prec(100) automatically calls mpfr::mpreal::set_default_prec(100) internally, but also sets the precision to the currently allocated mpfr::mpreal objects that are
    used as scratch space for lazy evaluation (see external/xdiff/external/lazy/README.md for more details).
    */
    lazy::set_default_mpreal_prec(100);
    run<lazy::LazyType<mpfr::mpreal>>("LazyType<mpfr::mpreal>");

    return 0;
}

/*
g++ -std=c++20 -O3 -DNDEBUG -march=native -Iinclude -Iexternal/xdiff/include -Iexternal/xdiff/external/lazy/include -Iexternal/polybox/include -Iexternal/ndspan/include tutorials/ArbitraryPrecision.cpp -o arbitrary_precision -lmpfr -lgmp && ./arbitrary_precision
*/