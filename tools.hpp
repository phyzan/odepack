#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <iomanip>

template<class Tt, class Ty>
using ode = Ty(*)(const Tt&, const Ty&, const std::vector<Tt>&);

template<class Tt, class Ty>
using event = std::function<bool(const Tt&, const Tt&, const Ty&, const Ty&)>;

template<class T>
std::vector<T> bisect(const T& a, const T& b, const T& xtol);


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

    void show() const{
        std::cout << std::endl << std::setprecision(15) << "t: " << t << "\ny: " << y << "\ndiverges: " << diverges << "\nis_stiff: " << is_stiff << "\nis_running: " << is_running << "\nUpdates: " << neval << std::endl;
    }
};

template<class Tt, class Ty>
struct ICS{
    Tt t0;
    Ty y0;
};


template<class Tt, class Ty>
struct OdeArgs{

    ICS<Tt, Ty> ics;
    Tt t;
    Tt h;
    Tt rtol = 1e-3;
    Tt atol = 1e-6;
    Tt cutoff_step = 0.;
    std::string method = "RK45";
    size_t max_frames = 0;
    std::vector<Tt> args;
    event<Tt, Ty> getcond = nullptr;
    event<Tt, Ty> breakcond = nullptr;
};


#endif