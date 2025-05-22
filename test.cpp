#include "src/variational.hpp"

const int N = -1;
using Tt = double;
using Ty = vec<Tt, N>;

void f(Ty& df, const Tt& t, const Ty& q, const std::vector<Tt>& args){
    df[0] = q[2];
    df[1] = q[3];
    df[2] = -q[0];
    df[3] = -q[1];
}

// Ty f(const Tt& t, const Ty& q, const std::vector<Tt>& args){
//     return {q[2], q[3], -q[0], -q[1]};
// }

// Ty f(const Tt& t, const Ty& q, const std::vector<Tt>& args){
//     Ty res(4);
//     res << q[2], q[3], -q[0], -q[1];
//     return res;
// }

Tt ps_func(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[1];
}

inline Tt stopfunc(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[1]-15;
}

bool check(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return q[3]>0;
}

int main(){
    Tt pi = 3.141592653589793238462;

    Tt t0 = 0;
    Ty q0(4);               // Allocate 4 elements
    q0 << 1., 1., 2.3, 4.5; // Use Eigen's stream operator
    Tt first_step = 0.;
    Tt rtol = 1e-16;
    Tt atol = 1e-12;
    Tt min_step = 0.;
    Tt max_step = 100;

    Tt tmax = 10001*pi/2;

    PreciseEvent<Tt, N> ps("Poincare Section", ps_func, check);
    // Event<Tt, Ty> ps("Poincare Section", ps_func, check);
    PeriodicEvent<Tt, N> ev2("periodic", 10, 1);
    // StopEvent<Tt, Ty> stopev("stop", stopfunc);
    // PeriodicEvent<Tt, Ty> ev3("periodic2", 10, 0.001);

    VariationalODE<Tt, N> ode(1000000, 0, f, t0, q0, rtol, atol, min_step, max_step, first_step, {}, "RK45", {});
    // ode.integrate(tmax).examine();
    // ode.var_integrate(tmax, 1000).examine();
    // ode.solver().state().show();
    // std::cout << ode.t_lyap().size() << std::endl;
    // ode.integrate(tmax, -1, {{"Poincare Section", 1}, {"periodic", 0}, {"Normalization", 0}}).examine();
    // std::cout << ode.runtime() << std::endl;
    // ode.solver().state().show();
    // ode.integrate(tmax, 100).examine();
    // ode.state().show();
    // ode.free();
    // while (true){
    //     ode.state().show();
    //     std::cin.get();
    //     ode.free();
    //     // ode.advance();
    //     ode.var_integrate(100, 1000);
    // }

    // ode.examine();
            //g++ -O3 -Wall -march=native -std=c++20 -fopenmp -fPIC test.cpp -o test -lmpfr -lgmp
    //sudo /opt/rocm-6.3.4/llvm/bin/clang -O3 -g -Wall -std=c++20 -fopenmp --offload=gfx1100 -fPIC test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    //g++ -O3 -Wall -std=c++20 -fopenmp -foffload=amdgcn-amdhsa -fno-lto -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp


    //sudo /opt/rocm-6.3.4/llvm/bin/clang -O3 -g -Wall -std=c++20 -fopenmp --offload-arch=gfx1100 -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    // sudo /opt/rocm-6.3.4/llvm/bin/clang++ -O3 -g -Wall -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1100 -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    //#pragma omp target teams distribute parallel for map(to: list_data[:s])

    //g++ -g -O3 -Wall -std=c++20 test.cpp -o test -lmpfr -lgmp

}