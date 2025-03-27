#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include <span>
#include "rk_adaptive.hpp"
#include <unordered_map>
#include <chrono>
#include <omp.h>


template<class Tt, class Ty>
OdeSolver<Tt, Ty>* getSolver(const SolverArgs<Tt, Ty>& S, const std::string& method) {

    OdeSolver<Tt, Ty>* solver = nullptr;

    if (method == "RK23") {
        solver = new RK23<Tt, Ty>(S);
    }
    else if (method == "RK45") {
        solver = new RK45<Tt, Ty>(S);
    }
    else {
        throw std::runtime_error("Unknown solver method: "+method);
    }

    return solver;
}

template<class Tt, class Ty>
class ODE{

public:

    ODE(ODE<Tt, Ty>&& other): _solver(other._solver), _t_arr(std::move(other._t_arr)), _q_arr(std::move(other._q_arr)), _Nevents(std::move(other._Nevents)), _runtime(other._runtime){
        other._solver = nullptr;
    }

    ODE(const SolverArgs<Tt, Ty>& S, const std::string& method) : _Nevents(S.events.size()){
        _solver = getSolver(S, method);
        _register_state();
    }

    ODE(const ODE<Tt, Ty>& other){
        _copy_data(other);
    }

    ODE(const Func<Tt, Ty> f, const Tt t0, const Ty q0, const Tt rtol, const Tt atol, const Tt min_step=0., const Tt& max_step=inf<Tt>(), const Tt first_step=0, const std::vector<Tt> args = {}, const std::string& method = "RK45", const std::vector<AnyEvent<Tt, Ty>*>& events = {}, const Func<Tt, Ty> mask=nullptr, const std::string& savedir="", const bool& save_events_only=false) : _Nevents(events.size()) {

        const SolverArgs<Tt, Ty> S = {f, t0, q0, rtol, atol, min_step, max_step, first_step, args, events, mask, savedir, save_events_only};
        _solver = getSolver(S, method);
        _register_state();
    }

    ODE<Tt, Ty>& operator=(const ODE<Tt, Ty>& other){
        if (&other == this) return *this;
        delete _solver;
        _copy_data(other);
        return *this;
    }

    ~ODE(){delete _solver;}

    const OdeResult<Tt, Ty> integrate(const Tt& interval, const int& max_frames=-1, const int& max_events=-1, const bool& terminate = true, const int& max_prints = 0, const bool& include_first=false);

    const OdeResult<Tt, Ty> go_to(const Tt& t, const int& max_frames=-1, const int& max_events=-1, const bool& terminate = true, const int& max_prints = 0, const bool& include_first=false);

    const SolverState<Tt, Ty> state() const {return _solver->state();}

    bool free(){
        return _solver->free();
    }

    bool resume(){
        return _solver->resume();
    }

    bool advance();

    const std::string& solver_filename() const{
        return _solver->filename();
    }

    std::map<std::string, std::vector<size_t>> event_map(const size_t& start_point=0) const{
        std::map<std::string, std::vector<size_t>> res;
        size_t index;
        for (size_t i=0; i<_solver->events_size(); i++){
            const AnyEvent<Tt, Ty>& ev = *_solver->event(i);
            res[ev.name()] = {};
            std::vector<size_t>& list = res[ev.name()];
            for (size_t j=0; j<_Nevents[i].size(); j++){
                index = _Nevents[i][j];
                if (index >= start_point){
                    list.push_back(index-start_point);
                }
            }
        }
        return res;
    }

    bool diverges()const{
        return _solver->diverges();
    }

    bool is_stiff()const{
        return _solver->is_stiff();
    }

    bool is_dead() const{
        return _solver->is_dead();
    }

    const std::vector<Tt>& t()const {return _t_arr;}
    const std::vector<Ty>& q()const{return _q_arr;}
    const double& runtime() const{return _runtime;}

    const OdeSolver<Tt, Ty>& solver() const{
        return *_solver;
    }

    bool save_data(const std::string& filename) const{
        if (typeid(Tt) != typeid(_q_arr[0][0])){
            std::cerr << ".save_data() only works for system of odes that are expressed in a 1D array\n";
            return false;
        }
        else{
            std::ofstream file(filename, std::ios::out);
            if (!file){
                std::cerr << "Could not open file:" << filename << "\n";
                return false;
            }

            for (size_t i = 0; i < _t_arr.size(); ++i) {
                write_chechpoint(file, _t_arr[i], _q_arr[i], _solver->current_event_index());
            }
            file.close(); // Close the file
            return true;
        }
    }

    void clear(){
        Tt t = _t_arr[_t_arr.size()-1];
        Ty q = _q_arr[_q_arr.size()-1];
        _t_arr.clear();
        _t_arr.shrink_to_fit();
        _t_arr.push_back(t);

        _q_arr.clear();
        _q_arr.shrink_to_fit();
        _q_arr.push_back(q);
        
        for (size_t i=0; i<_Nevents.size(); i++){
            _Nevents[i].clear();
            _Nevents[i].shrink_to_fit();
        }
    }

protected:

    OdeSolver<Tt, Ty>* _solver;
    std::vector<Tt> _t_arr;
    std::vector<Ty> _q_arr;
    std::vector<std::vector<size_t>> _Nevents;
    double _runtime = 0.;

private:
    void _register_state(){
        _t_arr.push_back(_solver->t());
        _q_arr.push_back(_solver->q());
    }

    void _copy_data(const ODE<Tt, Ty>& other){
        _solver = other._solver->clone();
        _t_arr = other._t_arr;
        _q_arr = other._q_arr;
        _Nevents = other._Nevents;
        _runtime = other._runtime;
    }

};

template<class Tt, class Ty>
void integrate_all(const std::vector<ODE<Tt, Ty>*>& list, const Tt& interval, const int& max_frames=-1, const int& max_events=-1, const bool& terminate=true, const int& threads=-1, const int& max_prints = 0){
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (ODE<Tt, Ty>* ode : list){
        ode->integrate(interval, max_frames, max_events, terminate, max_prints);
    }
}




/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/

template<class Tt, class Ty>
const OdeResult<Tt, Ty> ODE<Tt, Ty>::integrate(const Tt& interval, const int& max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
    return this->go_to(_solver->t()+interval, max_frames, max_events, terminate, max_prints, include_first);

}


template<class Tt, class Ty>
const OdeResult<Tt, Ty> ODE<Tt, Ty>::go_to(const Tt& t, const int& max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
    auto t1 = std::chrono::high_resolution_clock::now();
    const Tt t0 = _solver->t();
    const Tt interval = t-t0;
    const size_t N = _t_arr.size();
    long int event_counter = 0;
    long int frame_counter = 0;
    size_t i = N;
    const int MAX_PRINTS = max_prints;
    int prints = 0;
    _solver->reopen_file();
    _solver->set_goal(t);
    while (_solver->is_running()){
        if (_solver->advance()){
            if ((event_counter != max_events) && _solver->at_event()){
                _Nevents[_solver->current_event_index()].push_back(_t_arr.size());
                if (!_solver->current_event()->is_stop_event()){
                    if ( (++event_counter == max_events) && terminate){
                        _solver->stop("Max events reached");
                    }
                }
                ++i;
                _register_state();
            }
            else if ( (max_frames == -1) || (abs(_solver->t()-t0)*max_frames >= (frame_counter+1)*interval) ){
                _register_state();
                ++frame_counter;
                ++i;
            }
        }
        if (max_prints > 0){
            Tt percentage = (_solver->t() - t0)/(_solver->tmax()-t0);
            if (percentage*MAX_PRINTS >= prints){
                #pragma omp critical
                {
                    std::cout << std::setprecision(3) << "\033[2K\rProgress: " << 100*percentage << "%" <<   "    Events: " << event_counter << " / " << max_events << std::flush;
                    prints++;
                }

            }

        }
    }
    if (max_prints > 0){
        std::cout << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> rt = t2-t1;

    OdeResult<Tt, Ty> res = {subvec(_t_arr, N-include_first), subvec(_q_arr, N-include_first), event_map(N-include_first), _solver->diverges(), _solver->is_stiff(), !_solver->is_dead(), rt.count(), _solver->message()};

    _runtime += res.runtime;
    _solver->release_file();
    return res;
}

template<class Tt, class Ty>
bool ODE<Tt, Ty>::advance(){
    bool success = false;
    _solver->reopen_file();
    if (_solver->advance()){
        if (_solver->at_event()){
            _Nevents[_solver->current_event_index()].push_back(_t_arr.size());
        }
        _register_state();
        success = true;
    }
    _solver->release_file();
    return success;
}



#endif