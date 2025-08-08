#ifndef ODE_HPP
#define ODE_HPP

#include <cstddef>
#include <omp.h>
#include <stdexcept>
#include "tools.hpp"
#include "solvers.hpp"


struct EventOptions{
    std::string name;
    int max_events=-1;
    bool terminate=false;
    int period=1;
};



template<typename T, int N>
class EventCounter{

public:

    EventCounter(const std::vector<EventOptions>& options);

    DEFAULT_RULE_OF_FOUR(EventCounter);

    inline const int& operator[](const size_t& i) const;

    bool count_it(const size_t& i);

    inline const bool& is_running()const;

    inline bool can_fit(const size_t& event)const;

    inline const size_t& total()const;

private:

    std::vector<EventOptions> _options;
    std::vector<int> _counter;
    std::vector<int> _period_counter;
    size_t _total=0;
    bool _is_running = true;
};



template<typename T, int N>
class ODE{

public:

    ODE(const OdeSolver<T, N>& solver);

    ODE(ODE_CONSTRUCTOR(T, N));

    ODE(const ODE<T, N>& other);

    ODE(ODE<T, N>&& other);

    ODE<T, N>& operator=(const ODE<T, N>& other);

    ODE<T, N>& operator=(ODE<T, N>&& other);

    virtual ~ODE();

    virtual ODE<T, N>*                          clone() const;

    OdeSolution<T, N>                           rich_integrate(const T& interval, const std::vector<EventOptions>& event_options={}, const int& max_prints = 0);

    OdeResult<T, N>                             integrate(const T& interval, const int& max_frames=-1, const std::vector<EventOptions>& event_options={}, const int& max_prints = 0, const bool& include_first=false);

    OdeResult<T, N>                             go_to(const T& t, const int& max_frames=-1, const std::vector<EventOptions>& event_options={}, const int& max_prints = 0, const bool& include_first=false);

    SolverState<T, N>                           state() const;

    bool                                        free();

    bool                                        resume();

    bool                                        advance();

    std::map<std::string, std::vector<size_t>>  event_map(const size_t& start_point=0) const;

    std::vector<T>                              t_filtered(const std::string& event) const;

    std::vector<vec<T, N>>                      q_filtered(const std::string& event) const;

    bool                                        diverges()const;

    bool                                        is_dead() const;

    const std::vector<T>&                       t()const;

    const std::vector<vec<T, N>>&               q() const;

    const double&                               runtime() const;

    const OdeSolver<T, N>&                      solver() const;

    bool                                        save_data(const std::string& filename) const;

    virtual void                                clear();

    virtual void                                reset();

protected:

    OdeSolver<T, N>* _solver = nullptr;
    std::vector<T> _t_arr;
    std::vector<vec<T, N>> _q_arr;
    std::vector<std::vector<size_t>> _Nevents;
    double _runtime = 0;


    virtual void                _register_state(const int& event=-1);

private:

    void                        _copy_data(const ODE<T, N>& other);

    std::vector<EventOptions>   _validate_events(const std::vector<EventOptions>& options)const;

};

template<typename T, int N>
void integrate_all(const std::vector<ODE<T, N>*>& list, const T& interval, const int& max_frames=-1, const std::vector<EventOptions>& event_options ={}, const int& threads=-1, const bool& display_progress=false);




/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/



template<typename T, int N>
EventCounter<T, N>::EventCounter(const std::vector<EventOptions>& options) : _options(options), _counter(options.size(), 0), _period_counter(options.size(), 0) {
    for (size_t i=0; i<options.size(); i++){
        if (options[i].period < 1){
            throw std::runtime_error("The period argument in event options must be at least 1.");
        }
    }
}


template<typename T, int N>
inline const int& EventCounter<T, N>::operator[](const size_t& i) const{
    return _counter[i];
}

template<typename T, int N>
bool EventCounter<T, N>::count_it(const size_t& i){
    if (this->can_fit(i)){
        _period_counter[i]++;
        if (_period_counter[i] == _options[i].period){
            _period_counter[i] = 0;
            _counter[i]++;
            _total++;
            if ((_counter[i] == _options[i].max_events) && _options[i].terminate){
                _is_running = false;
            }
            return true;
        }
    }
    return false;
}

template<typename T, int N>
inline const bool& EventCounter<T, N>::is_running()const{
    return _is_running;
}

template<typename T, int N>
inline bool EventCounter<T, N>::can_fit(const size_t& event)const{
    return (_counter[event] != _options[event].max_events) && _is_running;
}


template<typename T, int N>
inline const size_t& EventCounter<T, N>::total()const{
    return _total;
}




template<typename T, int N>
ODE<T, N>::ODE(const OdeSolver<T, N>& solver) : _Nevents(solver.events().size()){
    _solver = solver.clone();
    _register_state();
}

template<typename T, int N>
ODE<T, N>::ODE(const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, const std::vector<T>& args, const std::vector<Event<T, N>*>& events, std::string method) : ODE(*get_solver(method, ARGS)){}


template<typename T, int N>
ODE<T, N>::ODE(const ODE& other){
    _copy_data(other);
}

template<typename T, int N>
ODE<T, N>::ODE(ODE&& other): _solver(other._solver), _t_arr(std::move(other._t_arr)), _q_arr(std::move(other._q_arr)), _Nevents(std::move(other._Nevents)), _runtime(other._runtime){
    other._solver = nullptr;
}

template<typename T, int N>
ODE<T, N>& ODE<T, N>::operator=(const ODE<T, N>& other){
    if (&other == this) return *this;
    _copy_data(other);
    return *this;
}

template<typename T, int N>
ODE<T, N>& ODE<T, N>::operator=(ODE<T, N>&& other){
    if (&other != this){
        _solver = other._solver;
        _t_arr = std::move(other._t_arr);
        _q_arr = std::move(other._q_arr);
        _Nevents = std::move(other._Nevents);
        _runtime = std::move(other._runtime);
        other._solver = nullptr;
    }
    return *this;
}

template<typename T, int N>
ODE<T, N>::~ODE(){
    delete _solver;
}


template<typename T, int N>
ODE<T, N>* ODE<T, N>::clone() const{
    return new ODE<T, N>(*this);
}

template<typename T, int N>
OdeSolution<T, N> ODE<T, N>::rich_integrate(const T& interval, const std::vector<EventOptions>& event_options, const int& max_prints){
    _solver->start_interpolation();
    OdeResult<T, N> res = this->integrate(interval, -1, event_options, max_prints, true);
    OdeSolution<T, N> rich_res(std::move(res), *_solver->interpolator());
    _solver->stop_interpolation();
    return rich_res;

}

template<typename T, int N>
OdeResult<T, N> ODE<T, N>::integrate(const T& interval, const int& max_frames, const std::vector<EventOptions>& event_options, const int& max_prints, const bool& include_first){
    return this->go_to(_solver->t()+interval, max_frames, event_options, max_prints, include_first);
}


template<typename T, int N>
OdeResult<T, N> ODE<T, N>::go_to(const T& t, const int& max_frames, const std::vector<EventOptions>& event_options, const int& max_prints, const bool& include_first){
    TimePoint t1 = now();
    const T t0 = _solver->t();
    const T interval = t-t0;
    const size_t Nt = _t_arr.size();
    long int frame_counter = 0;
    int prints = 0;

    _solver->set_goal(t);

    //check that all names in max_events are valid
    const std::vector<EventOptions> options = this->_validate_events(event_options);

    EventCounter<T, N> event_counter(options);

    while (_solver->is_running()){
        if (_solver->advance()){
            const int& ev = _solver->current_event_index();
            if (_solver->at_event() && event_counter.count_it(ev)){
                //the .count_it(ev) in the line above might have stopped the solver.
                //if the solver stopped for any other reason, that takes priority.
                //only if it is still running but max events have been reached, the solver will display "max events reached".
                if (_solver->is_running() && !event_counter.is_running()){
                    _solver->stop("Max events reached");
                }
                _register_state(ev);
            }
            else if ( (max_frames == -1) || (abs(_solver->t()-t0)*max_frames >= (frame_counter+1)*interval) ){
                _register_state();
                ++frame_counter;
            }
        }
        if (max_prints > 0){
            T percentage = (_solver->t() - t0)/(_solver->tmax()-t0);
            if (percentage*max_prints >= prints){
                #pragma omp critical
                {
                    std::cout << std::setprecision(std::log10(max_prints)+1) << "\033[2K\rProgress: " << 100*percentage << "%" <<   "    Events: " << event_counter.total() << std::flush;
                    prints++;
                }

            }

        }
    }
    if (max_prints > 0){
        std::cout << std::endl;
    }
    TimePoint t2 = now();

    OdeResult<T, N> res(subvec(_t_arr, Nt-include_first), subvec(_q_arr, Nt-include_first), event_map(Nt-include_first), _solver->diverges(), !_solver->is_dead(), as_duration(t1, t2), _solver->message());

    _runtime += res.runtime();
    return res;
}

template<typename T, int N>
SolverState<T, N> ODE<T, N>::state() const{
    return _solver->state();
}

template<typename T, int N>
bool ODE<T, N>::free(){
    return _solver->free();
}

template<typename T, int N>
bool ODE<T, N>::resume(){
    return _solver->resume();
}

template<typename T, int N>
bool ODE<T, N>::advance(){
    bool success = false;
    if (_solver->advance()){
        if (_solver->at_event()){
            _Nevents[_solver->current_event_index()].push_back(_t_arr.size());
        }
        _register_state(_solver->current_event_index());
        success = true;
    }
    return success;
}

template<typename T, int N>
std::map<std::string, std::vector<size_t>> ODE<T, N>::event_map(const size_t& start_point) const{
    std::map<std::string, std::vector<size_t>> res;
    size_t index;
    for (size_t i=0; i<_solver->events().size(); i++){
        const Event<T, N>& ev = _solver->events()[i];
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

template<typename T, int N>
std::vector<T> ODE<T, N>::t_filtered(const std::string& event) const {
    return _event_data(this->t(), this->event_map(), event);
}

template<typename T, int N>
std::vector<vec<T, N>> ODE<T, N>::q_filtered(const std::string& event) const {
    return _event_data(this->q(), this->event_map(), event);
}

template<typename T, int N>
bool ODE<T, N>::diverges() const{
    return _solver->diverges();
}

template<typename T, int N>
bool ODE<T, N>::is_dead() const{
    return _solver->is_dead();
}

template<typename T, int N>
const std::vector<T>& ODE<T, N>::t()const{
    return _t_arr;
}

template<typename T, int N>
const std::vector<vec<T, N>>& ODE<T, N>::q()const{
    return _q_arr;
}

template<typename T, int N>
const double& ODE<T, N>::runtime()const{
    return _runtime;
}

template<typename T, int N>
const OdeSolver<T, N>& ODE<T, N>::solver()const{
    return *_solver;
}

template<typename T, int N>
bool ODE<T, N>::save_data(const std::string& filename) const{
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

template<typename T, int N>
void ODE<T, N>::clear(){
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

template<typename T, int N>
void ODE<T, N>::reset(){
    _solver->reset();
    _t_arr.clear();
    _t_arr.shrink_to_fit();
    _q_arr.clear();
    _q_arr.shrink_to_fit();
    for (size_t i=0; i<_Nevents.size(); i++){
        _Nevents[i].clear();
        _Nevents[i].shrink_to_fit();
    }
    _runtime = 0;
    _register_state();
}

template<typename T, int N>
void ODE<T, N>::_register_state(const int& event){
    if (event != -1){
        _Nevents[event].push_back(_t_arr.size());
    }
    _t_arr.push_back(_solver->t());
    _q_arr.push_back(_solver->q());
}

template<typename T, int N>
void ODE<T, N>::_copy_data(const ODE<T, N>& other){
    delete _solver;
    _solver = other._solver->clone();
    _t_arr = other._t_arr;
    _q_arr = other._q_arr;
    _Nevents = other._Nevents;
    _runtime = other._runtime;
}

template<typename T, int N>
std::vector<EventOptions> ODE<T, N>::_validate_events(const std::vector<EventOptions>& options)const{
    std::vector<EventOptions> res(_solver->events().size());
    bool found;
    for (size_t i=0; i<options.size(); i++) {
        found = false;
        for (size_t j=0; j<_solver->events().size(); j++){
            if (_solver->events()[j].name() == options[i].name){
                found = true;
                break;
            }
        }
        if (!found){
            throw std::logic_error("Event name \""+options[i].name+"\" is invalid");
        }
    }

    for (size_t i=0; i<_solver->events().size(); i++){
        found = false;
        for (size_t j=0; j<options.size(); j++){
            if (options[j].name == _solver->events()[i].name()){
                found = true;
                res[i] = options[j];
                res[i].max_events = std::max(options[j].max_events, -1);
                break;
            }
        }
        if (!found){
            res[i] = {_solver->events()[i].name()};
        }
    }
    return res;
}


template<typename T, int N>
void integrate_all(const std::vector<ODE<T, N>*>& list, const T& interval, const int& max_frames, const std::vector<EventOptions>& event_options, const int& threads, const bool& display_progress){
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    int tot = 0;
    const int target = list.size();
    Clock clock;
    clock.start();
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (ODE<T, N>* ode : list){
        ode->integrate(interval, max_frames, event_options);
        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;
}


#endif