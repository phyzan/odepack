#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>


// USEFUL ALIASES

template<class Tt, size_t N=0>
using vec = std::conditional_t<(N == 0), Eigen::Array<Tt, Eigen::Dynamic, 1>, Eigen::Array<Tt, N, 1>>;

template<class Tt, class Ty>
using ode = Ty(*)(const Tt&, const Ty&, const std::vector<Tt>&);

template<class Tt, class Ty>
using ode_f = std::function<Ty(const Tt&, const Ty&, const std::vector<Tt>&)>;

template<class Tt, class Ty>
using event_f = std::function<bool(const Tt&, const Ty&, const Tt&, const Ty&)>;

template<class Tt, class Ty>
using event = bool(*)(const Tt&, const Ty&, const Tt&, const Ty&);

template<class Tt, class Ty, bool raw>
using ode_t = std::conditional_t<(raw==true), ode<Tt, Ty>, ode_f<Tt, Ty>>;

template<class Tt, class Ty, bool raw>
using event_t = std::conditional_t<(raw==true), event<Tt, Ty>, event_f<Tt, Ty>>;

template<class T, int Nr, int Nc>
T norm(const Eigen::Array<T, Nr, Nc>& f){
    return (f*f).sum();
}

template<class T, int Nr, int Nc>
Eigen::Array<T, Nr, Nc> cwise_abs(const Eigen::Array<T, Nr, Nc>& f){
    return f.cwiseAbs();
}

template<class T, int Nr, int Nc>
Eigen::Array<T, Nr, Nc> cwise_max(const Eigen::Array<T, Nr, Nc>& a, const Eigen::Array<T, Nr, Nc>& b){
    return a.cwiseMax(b);
}

template<class T>
T abs(const T& x){
    return (x > 0) ? x : -x;
}

template<class T, int Nr, int Nc>
bool All_isFinite(const Eigen::Array<T, Nr, Nc>& arr){
    return arr.isFinite().all();
}

template<class T, int Nr, int Nc>
std::vector<int> shape(const Eigen::Array<T, Nr, Nc>& arr){
    return {arr.rows(), arr.cols()};
}

template<class T>
std::vector<int> shape(const std::vector<T>& arr){
    return {arr.size()};
}

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

template<class Tt, class Ty>
struct OdeResult{

    std::vector<Tt> t;
    std::vector<Ty> y;
    bool diverges;
    bool is_stiff;
    long double runtime;
};



template<class Tt, class Ty>
struct SolverState{

    Tt t;
    Ty y;
    bool diverges;
    bool is_stiff;
    bool is_running;
    size_t neval;
    bool event;

    void show(const int& precision = 15) const{
        std::cout << std::endl << std::setprecision(precision) << "t: " << t << "\ny: " << y << "\ndiverges: " << diverges << "\nis_stiff: " << is_stiff << "\nis_running: " << is_running << "\nUpdates: " << neval << "\nevent: " << event << std::endl;
    }
};


template<class Tt, class Ty>
struct State{

    Tt t;
    Ty y;
    Tt dt;

};


template<class Tt, class Ty>
struct ICS{
    Tt t0;
    Ty y0;
};


template<class Tt, class Ty, bool raw_event>
struct OdeArgs{

    ICS<Tt, Ty> ics;
    Tt t;
    Tt h;
    Tt rtol = 1e-3;
    Tt atol = 1e-6;
    Tt cutoff_step = 0.;
    std::string method = "RK45";
    size_t max_frames = 0;
    std::vector<Tt> args = {};
    event_t<Tt, Ty, raw_event> getcond = nullptr;
    event_t<Tt, Ty, raw_event> breakcond = nullptr;

};



#endif