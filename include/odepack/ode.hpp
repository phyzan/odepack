#ifndef ODE_HPP
#define ODE_HPP


#include <omp.h>
#include "solvers.hpp"


struct EventOptions{
    std::string name;
    int max_events=-1;
    bool terminate=false;
    int period=1;
};



template<typename T, size_t N>
class EventCounter{

public:

    EventCounter(const std::vector<EventOptions>& options);

    DEFAULT_RULE_OF_FOUR(EventCounter);

    inline int operator[](size_t i) const;

    bool count_it(size_t i);

    inline bool is_running() const;

    inline bool can_fit(size_t event)const;

    inline size_t total()const;

private:

    std::vector<EventOptions> _options;
    std::vector<int> _counter;
    std::vector<int> _period_counter;
    size_t _total=0;
    bool _is_running = true;
};




template<typename T, size_t N>
class ODE{

public:

    ODE(ODE_CONSTRUCTOR(T, N));

    ODE(const ODE<T, N>& other);

    ODE(ODE<T, N>&& other) noexcept;

    ODE<T, N>& operator=(const ODE<T, N>& other);

    ODE<T, N>& operator=(ODE<T, N>&& other) noexcept;

    virtual ~ODE();

    virtual ODE<T, N>*                          clone() const;

    std::unique_ptr<ODE<T, N>>                  safe_clone() const;

    size_t                                      Nsys() const;

    OdeSolution<T, N>                           rich_integrate(const T& interval, const std::vector<EventOptions>& event_options={}, int max_prints = 0);

    OdeResult<T, N>                             integrate(const T& interval, int max_frames=-1, const std::vector<EventOptions>& event_options={}, int max_prints = 0, bool include_first=false);

    OdeResult<T, N>                             go_to(const T& t, int max_frames=-1, const std::vector<EventOptions>& event_options={}, int max_prints = 0, bool include_first=false);

    std::map<std::string, std::vector<size_t>>  event_map(size_t start_point=0) const;

    std::vector<T>                              t_filtered(const std::string& event) const;

    Array2D<T, 0, N>                            q_filtered(const std::string& event) const;

    bool                                        diverges() const;

    bool                                        is_dead() const;

    const std::vector<T>&                       t()const;

    const T&                                    t(size_t i) const;

    NdView<const T, Layout::C, 0, N>            q() const;

    NdView<const T, Layout::C, N>               q(size_t i) const;

    double                                      runtime() const;

    const OdeSolver<T, N>*                      solver() const;

    void                                        set_obj(const void* obj);

    virtual void                                clear();

    virtual void                                reset();

protected:

    ODE() = default;

    OdeSolver<T, N>* _solver = nullptr;
    std::vector<T> _t_arr;
    std::vector<T> _q_data;
    std::vector<std::vector<size_t>> _Nevents;
    double _runtime = 0;


    virtual void                                _register_state();

    virtual void                                _register_event(size_t i);

    void                                        _init(ODE_CONSTRUCTOR(T, N)){
        _Nevents = std::vector<std::vector<size_t>>(events.size());
        std::vector<const Event<T, N>*> new_evs(events.size()+1);
        TmaxEvent<T, N> tmax_ev;
        new_evs[0] = &tmax_ev;
        for (size_t i=0; i<events.size(); i++){
            new_evs[i+1] = events[i];
        }
        _solver = get_solver(method, ode, t0, q0, rtol, atol, min_step, max_step, first_step, args, new_evs).release();
        _register_state();
    }

private:

    void                                        _copy_data(const ODE<T, N>& other);

    std::vector<EventOptions>                   _validate_events(const std::vector<EventOptions>& options)const;

};


template<typename T, size_t N>
void integrate_all(const std::vector<ODE<T, N>*>& list, const T& interval, int max_frames, const std::vector<EventOptions>& event_options, int threads, bool display_progress){
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

template<typename T, size_t N>
EventCounter<T, N>::EventCounter(const std::vector<EventOptions>& options) : _options(options), _counter(options.size(), 0), _period_counter(options.size(), 0) {
    for (size_t i=0; i<options.size(); i++){
        if (options[i].period < 1){
            throw std::runtime_error("The period argument in event options must be at least 1.");
        }
    }
}


template<typename T, size_t N>
inline int EventCounter<T, N>::operator[](size_t i) const{
    return _counter[i];
}

template<typename T, size_t N>
bool EventCounter<T, N>::count_it(size_t i){
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

template<typename T, size_t N>
inline bool EventCounter<T, N>::is_running()const{
    return _is_running;
}

template<typename T, size_t N>
inline bool EventCounter<T, N>::can_fit(size_t event)const{
    return (_counter[event] != _options[event].max_events) && _is_running;
}


template<typename T, size_t N>
inline size_t EventCounter<T, N>::total()const{
    return _total;
}

template<typename T, size_t N>
ODE<T, N>::ODE(MAIN_CONSTRUCTOR(T, N), const std::string& method){
    _init(ARGS, method);
}

template<typename T, size_t N>
ODE<T, N>::ODE(const ODE& other){
    _copy_data(other);
}

template<typename T, size_t N>
ODE<T, N>::ODE(ODE&& other) noexcept: _solver(other._solver), _t_arr(std::move(other._t_arr)), _q_data(std::move(other._q_data)), _Nevents(std::move(other._Nevents)), _runtime(other._runtime){
    other._solver = nullptr;
}

template<typename T, size_t N>
ODE<T, N>& ODE<T, N>::operator=(const ODE<T, N>& other){
    if (&other == this) {return *this;}
    _copy_data(other);
    return *this;
}

template<typename T, size_t N>
ODE<T, N>& ODE<T, N>::operator=(ODE<T, N>&& other) noexcept{
    if (&other != this){
        delete _solver;
        _solver = other._solver;
        _t_arr = std::move(other._t_arr);
        _q_data = std::move(other._q_data);
        _Nevents = std::move(other._Nevents);
        _runtime = other._runtime;
        other._solver = nullptr;
    }
    return *this;
}

template<typename T, size_t N>
ODE<T, N>::~ODE(){
    delete _solver;
}

template<typename T, size_t N>
ODE<T, N>* ODE<T, N>::clone() const{
    return new ODE<T, N>(*this);
}

template<typename T, size_t N>
std::unique_ptr<ODE<T, N>> ODE<T, N>::safe_clone() const{
    return std::unique_ptr<ODE<T, N>>(this->clone());
}

template<typename T, size_t N>
size_t ODE<T, N>::Nsys() const{
    return _solver->Nsys();
}

template<typename T, size_t N>
OdeResult<T, N> ODE<T, N>::integrate(const T& interval, int max_frames, const std::vector<EventOptions>& event_options, int max_prints, bool include_first){
    if (interval < 0){
        throw std::runtime_error("Integration interval must be positive");
    }
    return this->go_to(_solver->t()+interval*_solver->direction(), max_frames, event_options, max_prints, include_first);
}

template<typename T, size_t N>
OdeSolution<T, N> ODE<T, N>::rich_integrate(const T& interval, const std::vector<EventOptions>& event_options, int max_prints){
    _solver->start_interpolation();
    OdeResult<T, N> res = this->integrate(interval, -1, event_options, max_prints, true);
    OdeSolution<T, N> rich_res(std::move(res), *_solver->interpolator());
    _solver->stop_interpolation();
    return rich_res;
}


template<typename T, size_t N>
OdeResult<T, N> ODE<T, N>::go_to(const T& t, int max_frames, const std::vector<EventOptions>& event_options, int max_prints, bool include_first){
    TimePoint t1 = now();
    const T t0 = _solver->t();
    if (t*_solver->direction() < t0*_solver->direction()){
        throw std::runtime_error("Integration can only advance along the initial direction");
    }
    const T interval = t-t0;
    const size_t Nt = _t_arr.size();
    long int frame_counter = 0;
    int prints = 0;
    size_t Nnew = 0;
    _solver->set_tmax(t);

    //check that all names in max_events are valid
    const std::vector<EventOptions> options = this->_validate_events(event_options);

    EventCounter<T, N> event_counter(options);

    while (_solver->is_running()){
        if (_solver->advance()){
            if (_solver->at_event()){
                for (const size_t& ev : _solver->event_col()){
                    if (ev > 0 && event_counter.count_it(ev-1)){
                        _register_event(ev-1);
                    }
                }
                //the .count_it(ev) in the line above might have stopped the solver.
                //if the solver stopped for any other reason, that takes priority.
                //only if it is still running but max events have been reached, the solver will display "max events reached".
                if (_solver->is_running() && !event_counter.is_running()){
                    _solver->stop("Max events reached");
                }
                _register_state();
                Nnew++;
            }
            else if ( (max_frames == -1) || (abs(_solver->t()-t0)*max_frames >= (frame_counter+1)*interval) ){
                ++frame_counter;
                _register_state();
                Nnew++;
            }
        }
        if (max_prints > 0){
            T percentage = (_solver->t() - t0)/(t-t0);
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

    OdeResult<T, N> res(subvec(_t_arr, Nt-static_cast<size_t>(include_first)), Array2D<T, 0, N>(_q_data.data()+(Nt-static_cast<size_t>(include_first))*_solver->Nsys(), Nnew, _solver->Nsys()), event_map(Nt-static_cast<size_t>(include_first)), _solver->diverges(), !_solver->is_dead(), as_duration(t1, t2), _solver->message());

    _runtime += res.runtime();
    return res;
}


template<typename T, size_t N>
std::map<std::string, std::vector<size_t>> ODE<T, N>::event_map(size_t start_point) const{
    std::map<std::string, std::vector<size_t>> res;
    size_t index;
    for (size_t i=1; i<_solver->event_col().size(); i++){
        const Event<T, N>& ev = _solver->event_col().event(i);
        res[ev.name()] = {};
        std::vector<size_t>& list = res[ev.name()];
        for (size_t j=0; j<_Nevents[i-1].size(); j++){
            index = _Nevents[i-1][j];
            if (index >= start_point){
                list.push_back(index-start_point);
            }
        }
    }
    return res;
}

template<typename T, size_t N>
std::vector<T> ODE<T, N>::t_filtered(const std::string& event) const {
    return _t_event_data(this->t().data(), this->event_map(), event);
}

template<typename T, size_t N>
Array2D<T, 0, N> ODE<T, N>::q_filtered(const std::string& event) const {
    return _q_event_data<T, N>(this->q().data(), this->event_map(), event, this->Nsys());
}

template<typename T, size_t N>
bool ODE<T, N>::diverges() const{
    return _solver->diverges();
}

template<typename T, size_t N>
bool ODE<T, N>::is_dead() const{
    return _solver->is_dead();
}

template<typename T, size_t N>
const std::vector<T>& ODE<T, N>::t()const{
    return _t_arr;
}

template<typename T, size_t N>
NdView<const T, Layout::C, 0, N> ODE<T, N>::q()const{
    return NdView<const T, Layout::C, 0, N>(_q_data.data(), _t_arr.size(), Nsys());
}

template<typename T, size_t N>
const T& ODE<T, N>::t(size_t i)const{
    return _t_arr[i];
}

template<typename T, size_t N>
NdView<const T, Layout::C, N> ODE<T, N>::q(size_t i)const{
    return NdView<const T, Layout::C, N>(_q_data.data() + i*this->Nsys(), this->Nsys());
}

template<typename T, size_t N>
double ODE<T, N>::runtime()const{
    return _runtime;
}

template<typename T, size_t N>
const OdeSolver<T, N>* ODE<T, N>::solver()const{
    return _solver;
}


template<typename T, size_t N>
void ODE<T, N>::set_obj(const void* obj){
    _solver->set_obj(obj);
}


template<typename T, size_t N>
void ODE<T, N>::clear(){
    T t = _t_arr[_t_arr.size()-1];
    _t_arr.clear();
    _t_arr.shrink_to_fit();
    _t_arr.push_back(t);

    _q_data = std::vector<T>(_q_data.begin()+(_t_arr.size()-1)*this->Nsys(), _q_data.end());

    for (size_t i=0; i<_Nevents.size(); i++){
        _Nevents[i].clear();
        _Nevents[i].shrink_to_fit();
    }
}

template<typename T, size_t N>
void ODE<T, N>::reset(){
    _solver->reset();
    _t_arr.clear();
    _t_arr.shrink_to_fit();
    _q_data.clear();
    _q_data.shrink_to_fit();
    for (size_t i=0; i<_Nevents.size(); i++){
        _Nevents[i].clear();
        _Nevents[i].shrink_to_fit();
    }
    _runtime = 0;
    _register_state();
}


template<typename T, size_t N>
void ODE<T, N>::_register_state(){
    _t_arr.push_back(_solver->t());
    _q_data.insert(_q_data.end(), _solver->q().begin(), _solver->q().end());
}

template<typename T, size_t N>
void ODE<T, N>::_register_event(size_t i){
    _Nevents[i].push_back(_t_arr.size());
}

template<typename T, size_t N>
void ODE<T, N>::_copy_data(const ODE<T, N>& other){
    delete _solver;
    _solver = other._solver->clone();
    _t_arr = other._t_arr;
    _q_data = other._q_data;
    _Nevents = other._Nevents;
    _runtime = other._runtime;
}



template<typename T, size_t N>
std::vector<EventOptions> ODE<T, N>::_validate_events(const std::vector<EventOptions>& options)const{

    //the check skips the first internal event which is about the tmax
    size_t Nevs = _solver->event_col().size();
    std::vector<EventOptions> res(Nevs-1);
    bool found;
    for (size_t i=0; i<options.size(); i++) {
        found = false;
        for (size_t j=1; j<Nevs; j++){
            if (_solver->event_col().event(j).name() == options[i].name){
                found = true;
                break;
            }
        }
        if (!found){
            throw std::logic_error("Event name \""+options[i].name+"\" is invalid");
        }
    }

    for (size_t i=0; i<Nevs-1; i++){ //the iteration skips the main TmaxEvent
        found = false;
        for (size_t j=0; j<options.size(); j++){
            if (options[j].name == _solver->event_col().event(i+1).name()){
                found = true;
                res[i] = options[j];
                res[i].max_events = std::max(options[j].max_events, -1);
                break;
            }
        }
        if (!found){
            res[i] = {_solver->event_col().event(i+1).name()};
        }
    }
    return res;
}



#endif