#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <iomanip>

template<class Tt, size_t N=0>
using vec = std::conditional_t<(N == 0), Eigen::Array<Tt, Eigen::Dynamic, 1>, Eigen::Array<Tt, N, 1>>;

template<class Tt, size_t N=0>
using ode = std::conditional_t<(N == 0), Eigen::Array<Tt, Eigen::Dynamic, 1>(*)(const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&, const std::vector<Tt>&), Eigen::Array<Tt, N, 1>(*)(const Tt&, const Eigen::Array<Tt, N, 1>&, const std::vector<Tt>&)>;


template<class Tt, size_t N=0>
using ode_t = std::function<std::conditional_t<(N == 0), Eigen::Array<Tt, Eigen::Dynamic, 1>(const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&, const std::vector<Tt>&), Eigen::Array<Tt, N, 1>(const Tt&, const Eigen::Array<Tt, N, 1>&, const std::vector<Tt>&)>>;

template<class Tt, size_t N=0>
using event = std::conditional_t<(N == 0), std::function<bool(const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&, const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&)>, std::function<bool(const Tt&, const Eigen::Array<Tt, N, 1>&, const Tt&, const Eigen::Array<Tt, N, 1>&)>>;

template<class Tt, size_t N=0>
using event_t = std::function<std::conditional_t<(N == 0), bool(const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&, const Tt&, const Eigen::Array<Tt, Eigen::Dynamic, 1>&), bool(const Tt&, const Eigen::Array<Tt, N, 1>&, const Tt&, const Eigen::Array<Tt, N, 1>&)>>;


template<class T>
std::vector<T> bisect(const T& a, const T& b, const T& xtol);


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

    void show() const{
        std::cout << std::endl << std::setprecision(15) << "t: " << t << "\ny: " << y << "\ndiverges: " << diverges << "\nis_stiff: " << is_stiff << "\nis_running: " << is_running << "\nUpdates: " << neval << std::endl;
    }
};

template<class Tt, size_t N>
struct ICS{
    Tt t0;
    vec<Tt, N> y0;
};


template<class Tt, size_t N>
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
    event<Tt, N> getcond = nullptr;
    event<Tt, N> breakcond = nullptr;
};


#endif