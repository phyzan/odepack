#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include <span>
#include <unordered_map>
#include <chrono>
#include <omp.h>
#include "solvers.hpp"


template<class T, int N>
class EventCounter{

public:

    EventCounter(const std::vector<int>& max_events) : _max_events(max_events), _counter(max_events.size(), 0) {}

    inline const int& operator[](const size_t& i) const{
        return _counter[i];
    }

    void count_it(const size_t& i){
        if (_counter[i] != _max_events[i]){
            _counter[i] ++;
            _total++;
        }
        else{
            throw std::runtime_error("Cannot register more events");
        }
    }

    bool still_counting()const{
        for (size_t i=0; i<_counter.size(); i++){
            if (_counter[i] != _max_events[i]){
                return true;
            }
        }
        return false;
    }

    inline bool can_fit(const size_t& event)const{
        return _counter[event] != _max_events[event];
    }

    inline const size_t& total()const{
        return _total;
    }

private:

    std::vector<int> _max_events;
    std::vector<int> _counter;
    size_t _total=0;
};



template<class T, int N>
class ODE{

public:

    ODE(ODE_CONSTRUCTOR(T, N)) : ODE(*get_solver(method, ARGS)){}

    ODE(ODE<T, N>&& other): _solver(other._solver), _t_arr(std::move(other._t_arr)), _q_arr(std::move(other._q_arr)), _Nevents(std::move(other._Nevents)), _runtime(other._runtime){
        other._solver = nullptr;
    }

    ODE(const ODE<T, N>& other){
        _copy_data(other);
    }

    ODE(const OdeSolver<T, N>& solver) : _Nevents(solver.events_size()){
        _solver = solver.clone();
        _register_state();
    }

    ODE<T, N>& operator=(const ODE<T, N>& other){
        if (&other == this) return *this;
        delete _solver;
        _copy_data(other);
        return *this;
    }

    virtual ~ODE(){delete _solver;}

    virtual ODE<T, N>* clone() const{
        return new ODE<T, N>(*this);
    }

    const OdeResult<T, N> integrate(const T& interval, const int& max_frames=-1, const std::map<std::string, int>& max_events={}, const int& max_prints = 0, const bool& include_first=false);

    const OdeResult<T, N> go_to(const T& t, const int& max_frames=-1, const std::map<std::string, int>& max_events={}, const int& max_prints = 0, const bool& include_first=false);

    const SolverState<T, N> state() const {return _solver->state();}

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
            const Event<T, N>& ev = *_solver->event(i);
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

    std::vector<T> t_filtered(const std::string& event) const {
        return _event_data(this->t(), this->event_map(), event);
    }

    std::vector<vec<T, N>> q_filtered(const std::string& event) const {
        return _event_data(this->q(), this->event_map(), event);
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

    const std::vector<T>& t()const {return _t_arr;}
    const std::vector<vec<T, N>>& q()const{return _q_arr;}
    const double& runtime() const{return _runtime;}

    const OdeSolver<T, N>& solver() const{
        return *_solver;
    }

    bool save_data(const std::string& filename) const{
        if (typeid(T) != typeid(_q_arr[0][0])){
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
                write_checkpoint(file, _t_arr[i], _q_arr[i], _solver->current_event_index());
            }
            file.close(); // Close the file
            return true;
        }
    }

    void clear(){
        T t = _t_arr[_t_arr.size()-1];
        vec<T, N> q = _q_arr[_q_arr.size()-1];
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

    OdeSolver<T, N>* _solver;
    std::vector<T> _t_arr;
    std::vector<vec<T, N>> _q_arr;
    std::vector<std::vector<size_t>> _Nevents;
    double _runtime = 0.;


    void _register_state(){
        _t_arr.push_back(_solver->t());
        _q_arr.push_back(_solver->q());
    }

private:
    void _copy_data(const ODE<T, N>& other){
        _solver = other._solver->clone();
        _t_arr = other._t_arr;
        _q_arr = other._q_arr;
        _Nevents = other._Nevents;
        _runtime = other._runtime;
    }

    void _assert_valid_event_map(const std::map<std::string, int>& m)const{
        bool found;
        for (std::map<std::string, int>::const_iterator it = m.begin(); it != m.end(); ++it) {
            found = false;
            std::string key = it->first;
            for (size_t j=0; j<_solver->events_size(); j++){
                if (_solver->event(j)->name() == key){
                    found = true;
                    break;
                }
            }
            if (!found){
                throw std::logic_error("Event name \""+key+"\" is invalid");
            }
        }
    }

};

template<class T, int N>
void integrate_all(const std::vector<ODE<T, N>*>& list, const T& interval, const int& max_frames=-1, const std::map<std::string, int>& max_events={}, const int& threads=-1, const int& max_prints = 0){
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (ODE<T, N>* ode : list){
        ode->integrate(interval, max_frames, max_events, max_prints);
    }
}




/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/

template<class T, int N>
const OdeResult<T, N> ODE<T, N>::integrate(const T& interval, const int& max_frames, const std::map<std::string, int>& max_events, const int& max_prints, const bool& include_first){
    return this->go_to(_solver->t()+interval, max_frames, max_events, max_prints, include_first);
}


template<class T, int N>
const OdeResult<T, N> ODE<T, N>::go_to(const T& t, const int& max_frames, const std::map<std::string, int>& max_events, const int& max_prints, const bool& include_first){
    auto t1 = std::chrono::high_resolution_clock::now();
    const T t0 = _solver->t();
    const T interval = t-t0;
    const size_t Nt = _t_arr.size();
    long int frame_counter = 0;
    size_t i = Nt;
    const int MAX_PRINTS = max_prints;
    int prints = 0;
    _solver->reopen_file();

    _solver->set_goal(t);

    //check that all names in max_events are valid
    _assert_valid_event_map(max_events);


    //manage max events
    std::vector<int> _max_ev(_solver->events_size());
    for (size_t i=0; i<_solver->events_size(); i++){
        _max_ev[i] = max_events.contains(_solver->event(i)->name()) ? std::max(max_events.at(_solver->event(i)->name()), -1) : -1;
    }
    EventCounter<T, N> event_counter(_max_ev);

    while (_solver->is_running()){
        if (_solver->advance()){
            const int& ev = _solver->current_event_index();
            if (_solver->at_event() && event_counter.can_fit(ev)){
                _Nevents[ev].push_back(_t_arr.size());
                event_counter.count_it(ev);
                if (!_solver->current_event()->is_stop_event()){
                    if (!event_counter.still_counting()){
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
            T percentage = (_solver->t() - t0)/(_solver->tmax()-t0);
            if (percentage*MAX_PRINTS >= prints){
                #pragma omp critical
                {
                    std::cout << std::setprecision(3) << "\033[2K\rProgress: " << 100*percentage << "%" <<   "    Events: " << event_counter.total() << std::flush;
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

    OdeResult<T, N> res = {subvec(_t_arr, Nt-include_first), subvec(_q_arr, Nt-include_first), event_map(Nt-include_first), _solver->diverges(), _solver->is_stiff(), !_solver->is_dead(), rt.count(), _solver->message()};

    _runtime += res.runtime;
    _solver->release_file();
    return res;
}

template<class T, int N>
bool ODE<T, N>::advance(){
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