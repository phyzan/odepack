#include <odepack/odepack.hpp>
#include <lazy/apps/mpfrLazy.hpp>

using namespace ode;
template<typename T>
void f(T* out, const T& /*t*/, const T* y, const T* /*args*/){
    out[0] = y[1];
    out[1] = -y[0];
}

void test_lazy_scalar(){
    using A = mpfr::mpreal;
    using T = A; //set T = A for performance comparison.

    // mpfr::mpreal::set_default_prec(256);
    lazy::set_default_mpreal_prec(256);
    auto y0 = std::vector<T>{1, -3};

    auto solver = getSolver<BDF, SolverPolicy::Static>(
        OdeData{.Rhs=f<T>}, T{0.}, View1D<T, 2>{y0.data()}, T{1e-40}, T{1e-40}
    );

    auto t_start = std::chrono::high_resolution_clock::now();
    solver.advance_until(0.001);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "q: " << solver.vector()[0] << " " << solver.vector()[1] << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms" << std::endl;
}