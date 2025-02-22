#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include "adaptive_rk.hpp"
#include <unordered_map>
#include <chrono>
#include <omp.h>

template<class Tt, size_t N, bool raw = true>
OdeSolver<Tt, N, raw>* getSolver(const ode_t<Tt, N, raw>& f, const OdeArgs<Tt, N, raw>& args) {

    SolverArgs<Tt, N, raw> S = to_SolverArgs(f, args);

    OdeSolver<Tt, N, raw>* solver = nullptr;

    RK23<Tt, N, raw> res1(S);
    RK45<Tt, N, raw> res2(S);

    if (args.method == "RK23") {
        solver = new RK23<Tt, N, raw>(S);
    }
    else if (args.method == "RK45") {
        solver = new RK45<Tt, N, raw>(S);
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

    const OdeResult<Tt, N> solve(const OdeArgs<Tt, N>& args) const;

    const std::vector<OdeResult<Tt, N>> solve_all(const std::vector<OdeArgs<Tt, N>>& args, int threads=-1) const;
};

template<class Tt, size_t N, bool raw_ode=true, bool raw_event=true>
struct OdeSet{

    ODE<Tt, N, raw_ode, raw_event> ode;
    OdeArgs<Tt, N> params;

};

template<class Tt, size_t N, bool raw_ode, bool raw_event>
std::vector<OdeResult<Tt, N>> dsolve_all(const std::vector<OdeSet<Tt, N, raw_ode, raw_event>>& data, int threads);


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/


template<class Tt, size_t N, bool raw_ode, bool raw_event>
const OdeResult<Tt, N> ODE<Tt, N, raw_ode, raw_event>::solve(const OdeArgs<Tt, N>& params) const{

    //extract data from params first
    const size_t& max_frames = params.max_frames;
    const event_t<Tt, N, raw_event>& getcond = params.getcond;
    const event_t<Tt, N, raw_event>& breakcond = params.breakcond;
    bool capture = false;
    size_t k=1;
    const Tt& t0 = params.ics.t0;
    const Tt& t_max = params.t;
    Tt t = t0;
    vec<Tt, N> y = params.ics.y0;
    
    Tt t_new;
    vec<Tt, N> y_new;
    std::vector<Tt> t_arr = {t};
    std::vector<vec<Tt, N>> y_arr = {y};

    OdeSolver<Tt, N, raw_ode>* solver = getSolver(this->f, params);

    auto t1 = std::chrono::high_resolution_clock::now();
    while (solver->is_running()){
        if (solver->advance()){
            t_new = solver->t_now();
            y_new = solver->y_now();

            if ( (breakcond != nullptr) && breakcond(t, y, t_new, y_new) ){
                
                //go back one step. This does not revert the stepsize of the solver,
                //but it doesnot matter as we will advance by an explicitly given step
                solver->set_ics(t, y);

                std::function<Tt(Tt)> func = [solver, t, y, breakcond](const Tt& _t) -> int {
                    vec<Tt, N> _y = solver->step(t, y, _t-t);
                    return (breakcond(t, y, _t, _y) > 0) ? 1: -1;
                };

                t_new = bisect(func, t, t_new, 1e-12)[2];
                solver->advance_by(t_new-t);
                y = solver->y_now();
                solver->stop();
            }
            else if ( (getcond != nullptr) && getcond(t, y, t_new, y_new) ){
                
                //go back one step. This does not revert the stepsize of the solver,
                //but it doesnot matter as we will advance by an explicitly given step
                solver->set_ics(t, y);

                std::function<Tt(Tt)> func = [solver, t, y, getcond](const Tt& _t) -> int {
                    vec<Tt, N> _y = solver->step(t, y, _t-t);
                    return (getcond(t, y, _t, _y) > 0) ? 1: -1;
                };

                t_new = bisect(func, t, t_new, 1e-12)[2];
                solver->advance_by(t_new-t);
                y = solver->y_now();
                capture = true;
            }
            else{
                y = y_new;
            }

            t = t_new;

            if (getcond != nullptr){
                if (capture){
                    t_arr.push_back(t);
                    y_arr.push_back(y);
                    capture = false;
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
const std::vector<OdeResult<Tt, N>> ODE<Tt, N, raw_ode, raw_event>::solve_all(const std::vector<OdeArgs<Tt, N>>& args, int threads) const{
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