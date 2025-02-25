#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include "rk_adaptive.hpp"
#include <unordered_map>
#include <chrono>
#include <omp.h>


template<class Tt, class Ty, bool raw_ode, bool raw_event>
OdeSolver<Tt, Ty, raw_ode, raw_event>* getSolver(const ode_t<Tt, Ty, raw_ode>& f, const OdeArgs<Tt, Ty, raw_event>& args) {

    SolverArgs<Tt, Ty, raw_ode, raw_event> S = to_SolverArgs<Tt, Ty, raw_ode, raw_event>(f, args);

    OdeSolver<Tt, Ty, raw_ode, raw_event>* solver = nullptr;

    if (args.method == "RK23") {
        solver = new RK23<Tt, Ty, raw_ode, raw_event>(S);
    }
    else if (args.method == "RK45") {
        solver = new RK45<Tt, Ty, raw_ode, raw_event>(S);
    }
    else {
        throw std::runtime_error("Unknown solver method");
    }

    return solver;
}

template<class Tt, class Ty, bool raw_ode = true, bool raw_event = true>
class ODE{

public:

    
    ode_t<Tt, Ty, raw_ode> f;

    ODE(ode_t<Tt, Ty, raw_ode> ode = nullptr) : f(ode) {};

    const OdeResult<Tt, Ty> solve(const OdeArgs<Tt, Ty, raw_event>& args) const;

    const std::vector<OdeResult<Tt, Ty>> solve_all(const std::vector<OdeArgs<Tt, Ty, raw_event>>& args, int threads=-1) const;
};

template<class Tt, class Ty, bool raw_ode, bool raw_event>
struct OdeSet{

    ODE<Tt, Ty, raw_ode, raw_event> ode;
    OdeArgs<Tt, Ty, raw_event> params;

};

template<class Tt, class Ty, bool raw_ode, bool raw_event>
std::vector<OdeResult<Tt, Ty>> dsolve_all(const std::vector<OdeSet<Tt, Ty, raw_ode, raw_event>>& data, int threads);


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/


template<class Tt, class Ty, bool raw_ode, bool raw_event>
const OdeResult<Tt, Ty> ODE<Tt, Ty, raw_ode, raw_event>::solve(const OdeArgs<Tt, Ty, raw_event>& params) const{

    auto t1 = std::chrono::high_resolution_clock::now();
    //MAYBE KEEP ALL WHEN EVENTS. IF SO, RETURN ARRAY WITH EVENTS TOO

    //extract data from params first
    const size_t& max_frames = params.max_frames;
    const Tt& t0 = params.ics.t0;
    const Tt& t_max = params.t;
    Tt t = t0;
    Ty y = params.ics.y0;
    size_t k=1;
    std::vector<Tt> t_arr;
    std::vector<Ty> y_arr;

    
    if (max_frames > 1){
        t_arr.reserve(max_frames);
        y_arr.reserve(max_frames);
    }
    t_arr.push_back(t);
    y_arr.push_back(y);

    OdeSolver<Tt, Ty, raw_ode, raw_event>* solver = getSolver<Tt, Ty, raw_ode, raw_event>(this->f, params);

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
            else if ( (max_frames == 0) || (abs(t-t0)*(max_frames-1) >= k*abs(t_max-t0)) ){
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

    SolverState<Tt, Ty> state = solver->state();

    OdeResult<Tt, Ty> res{t_arr, y_arr, state.diverges, state.is_stiff, runtime.count()};

    delete solver;
    solver = nullptr;

    return res;
}


template<class Tt, class Ty, bool raw_ode, bool raw_event>
const std::vector<OdeResult<Tt, Ty>> ODE<Tt, Ty, raw_ode, raw_event>::solve_all(const std::vector<OdeArgs<Tt, Ty, raw_event>>& args, int threads) const{
    //define all sets of ode-args
    std::vector<OdeSet<Tt, Ty, raw_ode, raw_event>> data(args.size());
    for (size_t i=0; i<args.size(); i++){
        data[i] = {*this, args[i]};
    }

    std::vector<OdeResult<Tt, Ty>> res = dsolve_all(data, threads);

    return res;
}

template<class Tt, class Ty, bool raw_ode, bool raw_event>
std::vector<OdeResult<Tt, Ty>> dsolve_all(const std::vector<OdeSet<Tt, Ty, raw_ode, raw_event>>& data, int threads){

    const size_t n = data.size();
    std::vector<OdeResult<Tt, Ty>> res(n);

    threads = (threads == -1) ? omp_get_max_threads() : threads;
    if (threads > 1){
        #pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (size_t i=0; i<n; i++){
            res[i] = data[i].ode.solve(data[i].params);
        }
    }
    else{
        for (size_t i=0; i<n; i++){
            res[i] = data[i].ode.solve(data[i].params);
        }
    }

    return res;
}


#endif