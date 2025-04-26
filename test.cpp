#include "src/ode.hpp"

using Tt = double;
using Ty = vec<Tt, 4>;

Ty f(const Tt& t, const Ty& q, const std::vector<Tt>& args){
    return {q[2], q[3], -q[0], -q[1]};
}

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
    Ty q0 = {1., 1., 2.3, 4.5};
    Tt first_step = 0.;
    Tt rtol = 1e-16;
    Tt atol = 1e-12;
    Tt min_step = 0.;
    Tt max_step = 100;

    Tt tmax = 10001*pi/2;

    // Event<Tt, Ty> ps("Poincare Section", ps_func, check);
    // PeriodicEvent<Tt, Ty> ev2("periodic", 10, 0);
    // StopEvent<Tt, Ty> stopev("stop", stopfunc);
    // PeriodicEvent<Tt, Ty> ev3("periodic2", 10, 0.001);

    // ODE<Tt, Ty> ode(f, t0, q0, rtol, atol, min_step, max_step, first_step, {}, "RK45", 1e-10);
    // ode.integrate(tmax, 100).examine();
    // ode.state().show();
    // ode.free();
    // while (true){
    //     ode.state().show();
    //     std::cin.get();
    //     ode.free();
    //     ode.advance();
    // }

    // ode.examine();
            //g++ -O3 -Wall -march=native -std=c++20 -fopenmp -fPIC test.cpp -o test -lmpfr -lgmp
    //sudo /opt/rocm-6.3.4/llvm/bin/clang -O3 -g -Wall -std=c++20 -fopenmp --offload=gfx1100 -fPIC test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    //g++ -O3 -Wall -std=c++20 -fopenmp -foffload=amdgcn-amdhsa -fno-lto -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp


    //sudo /opt/rocm-6.3.4/llvm/bin/clang -O3 -g -Wall -std=c++20 -fopenmp --offload-arch=gfx1100 -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    // sudo /opt/rocm-6.3.4/llvm/bin/clang++ -O3 -g -Wall -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1100 -fPIC -fcf-protection=none -fno-stack-protector test.cpp -o test -lmpfr -lgmp -lstdc++ -lm

    //#pragma omp target teams distribute parallel for map(to: list_data[:s])

}