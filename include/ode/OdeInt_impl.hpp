#ifndef ODE_INT_IMPL_HPP
#define ODE_INT_IMPL_HPP

#include "OdeInt.hpp"
#include "Tools_impl.hpp"

namespace ode{


// EventCounter implementations
template<typename T, size_t N>
EventCounter<T, N>::EventCounter(const std::vector<EventOptions>& options) : _options(options), _counter(options.size(), 0), _period_counter(options.size(), 0) {
    for (size_t i=0; i<options.size(); i++){
        if (options[i].period < 1){
            throw std::runtime_error("The period argument in event options must be at least 1.");
        }
    }
}

template<typename T, size_t N>
 int EventCounter<T, N>::operator[](size_t i) const{
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
bool EventCounter<T, N>::is_running()const{
    return _is_running;
}

template<typename T, size_t N>
bool EventCounter<T, N>::can_fit(size_t event)const{
    return (_counter[event] != _options[event].max_events) && _is_running;
}

template<typename T, size_t N>
size_t EventCounter<T, N>::total()const{
    return _total;
}

// ODE implementations
template<typename T, size_t N>
template<typename RhsType, typename JacType>
ODE<T, N>::ODE(MAIN_CONSTRUCTOR(T), EVENTS events, Integrator method){
    _init(ARGS, events, method);
}

template<typename T, size_t N>
void ODE<T, N>::Rhs(T* out, const T& t, const T* q) const{
    _solver->Rhs(out, t, q);
}

template<typename T, size_t N>
void ODE<T, N>::Jac(T* out, const T& t, const T* q, const T* dt) const{
    _solver->Jac(out, t, q, dt);
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
OdeResult<T, N> ODE<T, N>::integrate(const T& interval, const std::vector<T>& t_array, const std::vector<EventOptions>& event_options, int max_prints){
    if (interval < 0){
        throw std::runtime_error("Integration interval must be positive");
    }
    return this->priv_integrate_until(_solver->t()+interval*_solver->direction(), t_array, event_options, max_prints);
}

template<typename T, size_t N>
OdeResult<T, N> ODE<T, N>::integrate(const T& interval, const std::vector<EventOptions>& event_options, int max_prints){
    if (interval < 0){
        throw std::runtime_error("Integration interval must be positive");
    }
    return this->priv_integrate_until(_solver->t()+interval*_solver->direction(), EmptyArr<T>{}, event_options, max_prints);
}

template<typename T, size_t N>
OdeResult<T, N> ODE<T, N>::integrate_until(const T& t, const std::vector<EventOptions>& event_options, int max_prints){
    return this->priv_integrate_until(t, EmptyArr<T>{}, event_options, max_prints);
}

template<typename T, size_t N>
OdeResult<T, N> ODE<T, N>::integrate_until(const T& t, const std::vector<T>& t_eval, const std::vector<EventOptions>& event_options, int max_prints){
    return this->priv_integrate_until(t, t_eval, event_options, max_prints);
}

template<typename T, size_t N>
OdeSolution<T, N> ODE<T, N>::rich_integrate(const T& interval, const std::vector<EventOptions>& event_options, int max_prints){
    _solver->start_interpolation();
    OdeResult<T, N> res = this->integrate(interval, event_options, max_prints);
    OdeSolution<T, N> rich_res(std::move(res), *_solver->interpolator());
    _solver->stop_interpolation();
    return rich_res;
}

template<typename T, size_t N>
template<typename ArrayType>
OdeResult<T, N> ODE<T, N>::priv_integrate_until(const T& t_max, const ArrayType& t_store, const std::vector<EventOptions>& event_options, int max_prints){
    if (_solver->is_dead()){
        return OdeResult<T, N>({}, {}, {}, _solver->diverges(), false, 0, _solver->status());
    }else if (t_max*_solver->direction() < _solver->t()*_solver->direction()){
        return OdeResult<T, N>({}, {}, {}, false, false, 0, "Cannot integrate in opposite direction"); //cannot integrate in opposite direction
    }
    TimePoint TIME_START = Clock::now();
    constexpr bool store_explicit_steps = !std::is_same_v<std::decay_t<ArrayType>, EmptyArr<T>>;
    // ------------------------------ IMPLEMENTATION --------------------------------------
    _solver->resume();
    const T         t0 = _solver->t();
    const bool      first_eval_t0 = (t_store.size() > 0 && t_store[0] == t0);
    const bool      include_first = (!store_explicit_steps || first_eval_t0);
    const size_t    t_start_idx = _t_arr.size() - include_first;
    int             prints = 0;
    size_t          n_points = include_first;
    const char*     terminate_message = nullptr;
    View1D<T>       t_eval(t_store.data() + first_eval_t0, t_store.size() - first_eval_t0);

    //check that all names in max_events are valid
    const std::vector<EventOptions> options = this->_validate_events(event_options);
    EventCounter<T, N>              event_counter(options);

    auto event_state_valid = [&]()LAMBDA_INLINE{
        bool res = false;
        for (const size_t& ev : _solver->event_col()){
            if (event_counter.count_it(ev)){
                res = true;
                _register_event(ev);
            }
        }
        return res;
    };

    // Since we pass an array of t_eval in the solver later, if step_ptr is not null,
    // it is guaranteed to point to an element in t_eval.
    auto observer = [&](const T& t, const T* q, const T* step_ptr) -> void {
        const bool at_valid_event = _solver->at_event() && event_state_valid();

        if constexpr (!store_explicit_steps) {
            _register_state();
            n_points++;
        } else if (at_valid_event || step_ptr) {
            _register_state();
            n_points++;
        }

        if (at_valid_event && !event_counter.is_running()) {
            terminate_message = "Max events reached";
        }

        if (_solver->t() == t_max){
            terminate_message = "t-goal";
        }

        if (terminate_message && _solver->is_running()){
            _solver->stop(terminate_message);
        }

        // =========================== Manage console output ==========================
        if (max_prints > 0){
            T percentage = (_solver->t() - t0)/(t_max-t0);
            if (percentage*max_prints >= prints){
                #pragma omp critical
                {
                    std::cout << std::setprecision(std::log10(max_prints)+1) << "\033[2K\rProgress: " << 100*percentage << "%" <<   "    Events: " << event_counter.total() << std::flush;
                    prints++;
                }

            }

        }
        // ============================================================================
    };

    _solver->observe_until(t_max, observer, t_eval);

    if (max_prints > 0){
        std::cout << std::endl;
    }
    TimePoint TIME_END = Clock::now();
    OdeResult<T, N> res(subvec(_t_arr, t_start_idx, n_points), Array2D<T, 0, N>(_q_data.data()+t_start_idx*_solver->Nsys(), n_points, _solver->Nsys()), event_map(t_start_idx), _solver->diverges(), !_solver->is_dead(), Clock::as_duration(TIME_START, TIME_END), _solver->status());
    _runtime += res.runtime();
    return res;
}

template<typename T, size_t N>
std::map<std::string, std::vector<size_t>> ODE<T, N>::event_map(size_t start_point) const{
    std::map<std::string, std::vector<size_t>> res;
    size_t index;
    for (size_t i=0; i<_solver->event_col().size(); i++){
        const Event<T>& ev = _solver->event_col().event(i);
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
View<T, Layout::C, 0, N> ODE<T, N>::q()const{
    return View<T, Layout::C, 0, N>(_q_data.data(), _t_arr.size(), Nsys());
}

template<typename T, size_t N>
const T& ODE<T, N>::t(size_t i)const{
    return _t_arr[i];
}

template<typename T, size_t N>
View<T, Layout::C, N> ODE<T, N>::q(size_t i)const{
    return View<T, Layout::C, N>(_q_data.data() + i*this->Nsys(), this->Nsys());
}

template<typename T, size_t N>
double ODE<T, N>::runtime()const{
    return _runtime;
}

template<typename T, size_t N>
const OdeRichSolver<T, N>* ODE<T, N>::solver()const{
    return _solver.ptr();
}

template<typename T, size_t N>
void ODE<T, N>::set_obj(const void* obj){
    _solver->set_obj(obj);
}

template<typename T, size_t N>
void ODE<T, N>::clear(){
    T t = _t_arr.back();
    _t_arr.clear();
    _t_arr.shrink_to_fit();
    _t_arr.push_back(t);

    _q_data = std::vector<T>(_q_data.begin()+(_t_arr.size()-1)*this->Nsys(), _q_data.end());

    for (auto & _Nevent : _Nevents){
        _Nevent.clear();
        _Nevent.shrink_to_fit();
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
template<typename RhsType, typename JacType>
void ODE<T, N>::_init(MAIN_CONSTRUCTOR(T), EVENTS events, Integrator method){
    _solver.take_ownership(get_virtual_solver<T, N>(method, ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events).release());
    _Nevents = std::vector<std::vector<size_t>>(events.size());
    _register_state();
}

template<typename T, size_t N>
void ODE<T, N>::_register_state(){
    _t_arr.push_back(_solver->t());
    _q_data.insert(_q_data.end(), _solver->vector().begin(), _solver->vector().end());
}

template<typename T, size_t N>
void ODE<T, N>::_register_event(size_t i){
    _Nevents[i].push_back(_t_arr.size());
}


template<typename T, size_t N>
std::vector<EventOptions> ODE<T, N>::_validate_events(const std::vector<EventOptions>& options)const{


    size_t Nevs = _solver->event_col().size();
    std::vector<EventOptions> res(Nevs);
    bool found;
    for (size_t i=0; i<options.size(); i++) {
        found = false;
        for (size_t j=0; j<Nevs; j++){
            if (_solver->event_col().event(j).name() == options[i].name){
                found = true;
                break;
            }
        }
        if (!found){
            throw std::logic_error("Event name \""+options[i].name+"\" is invalid");
        }
    }

    for (size_t i=0; i<Nevs; i++){
        found = false;
        for (const auto& option : options){
            if (option.name == _solver->event_col().event(i).name()){
                found = true;
                res[i] = option;
                res[i].max_events = ndspan::max(option.max_events, -1);
                break;
            }
        }
        if (!found){
            res[i] = {_solver->event_col().event(i).name()};
        }
    }
    return res;
}



} // namespace ode

#endif // ODE_INT_IMPL_HPP