#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <iomanip>
#include <mpfr.h>


// USEFUL ALIASES

template<class Tt, size_t N=0>
using vec = std::conditional_t<(N == 0), Eigen::Array<Tt, Eigen::Dynamic, 1>, Eigen::Array<Tt, N, 1>>;

template<class Tt, size_t N=0>
using ode = vec<Tt, N>(*)(const Tt&, const vec<Tt, N>&, const std::vector<Tt>&);

template<class Tt, size_t N=0>
using ode_f = std::function<vec<Tt, N>(const Tt&, const vec<Tt, N>&, const std::vector<Tt>&)>;

template<class Tt, size_t N=0>
using event_f = std::function<bool(const Tt&, const vec<Tt, N>&, const Tt&, vec<Tt, N>&)>;

template<class Tt, size_t N=0>
using event = bool(*)(const Tt&, const vec<Tt, N>&, const Tt&, const vec<Tt, N>&);

template<class Tt, size_t N=0, bool raw = true>
using ode_t = std::conditional_t<(raw==true), ode<Tt, N>, ode_f<Tt, N>>;

template<class Tt, size_t N=0, bool raw = true>
using event_t = std::conditional_t<(raw==true), event<Tt, N>, event_f<Tt, N>>;


//BISECTION USED FOR EVENTS IN ODES

template<class T, typename Callable>
std::vector<T> bisect(Callable&& f, const T& a, const T& b, const T& xtol){
    T err = 2*xtol;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    if (f(a)*f(b) > 0){
        throw std::runtime_error("Root not bracketed");
    }

    while (err > xtol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            break;
        }
        fm = f(c);
        if (f(_a) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = abs(_b-_a);
    }
    return {_a, c, _b};
}


//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<class Tt, size_t N>
struct OdeResult{

    std::vector<Tt> t;
    std::vector<vec<Tt, N>> y;
    bool diverges;
    bool is_stiff;
    long double runtime;
};



template<class Tt, size_t N>
struct SolverState{

    Tt t;
    vec<Tt, N> y;
    bool diverges;
    bool is_stiff;
    bool is_running;
    size_t neval;
    bool event;

    void show(const int& precision = 15) const{
        std::cout << std::endl << std::setprecision(precision) << "t: " << t << "\ny: " << y.transpose() << "\ndiverges: " << diverges << "\nis_stiff: " << is_stiff << "\nis_running: " << is_running << "\nUpdates: " << neval << "\nevent: " << event << std::endl;
    }
};


template<class Tt, size_t N>
struct State{

    Tt t;
    vec<Tt, N> y;
    Tt dt;

};


template<class Tt, size_t N>
struct ICS{
    Tt t0;
    vec<Tt, N> y0;
};


template<class Tt, size_t N, bool raw_event>
struct OdeArgs{

    ICS<Tt, N> ics;
    Tt t;
    Tt h;
    Tt rtol = 1e-3;
    Tt atol = 1e-6;
    Tt cutoff_step = 0.;
    std::string method = "RK45";
    size_t max_frames = 0;
    std::vector<Tt> args;
    event_t<Tt, N, raw_event> getcond = nullptr;
    event_t<Tt, N, raw_event> breakcond = nullptr;

};



#endif