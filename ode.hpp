#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include "adaptive_rk.hpp"
#include <unordered_map>
#include <chrono>
#include <omp.h>

template<class Tt, size_t N, bool raw_ode, bool raw_event>
OdeSolver<Tt, N, raw_ode, raw_event>* getSolver(const ode_t<Tt, N, raw_ode>& f, const OdeArgs<Tt, N, raw_event>& args) {

    SolverArgs<Tt, N, raw_ode, raw_event> S = to_SolverArgs<Tt, N, raw_ode, raw_event>(f, args);

    OdeSolver<Tt, N, raw_ode, raw_event>* solver = nullptr;

    if (args.method == "RK23") {
        solver = new RK23<Tt, N, raw_ode, raw_event>(S);
    }
    else if (args.method == "RK45") {
        solver = new RK45<Tt, N, raw_ode, raw_event>(S);
    }
    else {
        throw std::runtime_error("Unknown solver method");
    }

    return solver;
}

template<class Tt, size_t N = 0, bool raw_ode = true, bool raw_event = true>
class ODE{

public:

    
    ode_t<Tt, N, raw_ode> f;

    ODE(ode_t<Tt, N, raw_ode> ode = nullptr) : f(ode) {};

    const OdeResult<Tt, N> solve(const OdeArgs<Tt, N, raw_event>& args) const;

    const std::vector<OdeResult<Tt, N>> solve_all(const std::vector<OdeArgs<Tt, N, raw_event>>& args, int threads=-1) const;
};

template<class Tt, size_t N, bool raw_ode, bool raw_event>
struct OdeSet{

    ODE<Tt, N, raw_ode, raw_event> ode;
    OdeArgs<Tt, N, raw_event> params;

};

template<class Tt, size_t N, bool raw_ode, bool raw_event>
std::vector<OdeResult<Tt, N>> dsolve_all(const std::vector<OdeSet<Tt, N, raw_ode, raw_event>>& data, int threads);


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/


template<class Tt, size_t N, bool raw_ode, bool raw_event>
const OdeResult<Tt, N> ODE<Tt, N, raw_ode, raw_event>::solve(const OdeArgs<Tt, N, raw_event>& params) const{


    // PREALLOCATE WHEN GIVEN MAX_FRAMES
    //CHECK WITH k and MAX_FRAMES AND ALL TO MAKE SURE ALL IS GOOD
    //MAYBE KEEP ALL WHEN EVENTS

    //extract data from params first
    const size_t& max_frames = params.max_frames;
    size_t k=1;
    const Tt& t0 = params.ics.t0;
    const Tt& t_max = params.t;
    Tt t = t0;
    vec<Tt, N> y = params.ics.y0;

    std::vector<Tt> t_arr = {t};
    std::vector<vec<Tt, N>> y_arr = {y};

    OdeSolver<Tt, N, raw_ode, raw_event>* solver = getSolver<Tt, N, raw_ode, raw_event>(this->f, params);

    auto t1 = std::chrono::high_resolution_clock::now();
    while (solver->is_running()){
        if (solver->advance()){
            t = solver->t_now();
            y = solver->y_now();
            if (solver->getevent != nullptr){
                if (solver->event()){
                    t_arr.push_back(t);
                    y_arr.push_back(y);
                    ++k;
                }
            }
            else if ( (max_frames == 0) || (std::abs(t-t0)*(max_frames-1) >= k*std::abs(t_max-t0)) ){
                t_arr.push_back(t);
                y_arr.push_back(y);
                ++k;
            }

            if (k == max_frames){
                solver->stop();
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<long double> runtime = t2-t1;

    SolverState<Tt, N> state = solver->state();

    OdeResult<Tt, N> res{t_arr, y_arr, state.diverges, state.is_stiff, runtime.count()};

    delete solver;
    solver = nullptr;

    return res;
}


template<class Tt, size_t N, bool raw_ode, bool raw_event>
const std::vector<OdeResult<Tt, N>> ODE<Tt, N, raw_ode, raw_event>::solve_all(const std::vector<OdeArgs<Tt, N, raw_event>>& args, int threads) const{
    //define all sets of ode-args
    std::vector<OdeSet<Tt, N, raw_ode, raw_event>> data(args.size());
    for (size_t i=0; i<args.size(); i++){
        data[i] = {*this, args[i]};
    }

    std::vector<OdeResult<Tt, N>> res = dsolve_all(data, threads);

    return res;
}

template<class Tt, size_t N, bool raw_ode, bool raw_event>
std::vector<OdeResult<Tt, N>> dsolve_all(const std::vector<OdeSet<Tt, N, raw_ode, raw_event>>& data, int threads){

    const size_t n = data.size();
    std::vector<OdeResult<Tt, N>> res(n);

    threads = (threads == -1) ? omp_get_max_threads() : threads;
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t i=0; i<n; i++){
        res[i] = data[i].ode.solve(data[i].params);
    }

    return res;
}


#endif