#ifndef ODE_INT_IMPL_HPP
#define ODE_INT_IMPL_HPP

#include <algorithm>

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
ODE<T, N>::ODE(MAIN_CONSTRUCTOR(T), EVENTS events, Integrator method) : ODE(nsys){
    init(ARGS, events, method);
}

template<typename T, size_t N>
ODE<T, N>::ODE(size_t nsys) : event_data_(nsys){
    orbit_data_.nsys = nsys;
}

template<typename T, size_t N>
void ODE<T, N>::Rhs(T* out, const T& t, const T* q) const{
    solver_->Rhs(out, t, q);
}

template<typename T, size_t N>
void ODE<T, N>::Jac(T* out, const T& t, const T* q, const T* dt) const{
    solver_->Jac(out, t, q, dt);
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
    return solver_->Nsys();
}

template<typename T, size_t N>
template<typename Callable>
bool ODE<T, N>::integrate(OdeResult<T, N>* out, const T& interval, const std::vector<T>& t_array, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    if (interval < 0){
        throw std::runtime_error("Integration interval must be positive");
    }
    return this->priv_integrate_until(out, solver_->t()+interval*solver_->direction(), t_array, event_options, std::forward<Callable>(observer), max_prints);
}

template<typename T, size_t N>
template<typename Callable>
bool ODE<T, N>::integrate(OdeResult<T, N>* out, const T& interval, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    if (interval < 0){
        throw std::runtime_error("Integration interval must be positive");
    }
    return this->priv_integrate_until(out, solver_->t()+interval*solver_->direction(), EmptyArr<T>{}, event_options, std::forward<Callable>(observer), max_prints);
}

template<typename T, size_t N>
template<typename Callable>
bool ODE<T, N>::integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    return this->priv_integrate_until(out, t, EmptyArr<T>{}, event_options, std::forward<Callable>(observer), max_prints);
}

template<typename T, size_t N>
template<typename Callable>
bool ODE<T, N>::integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<T>& t_eval, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    return this->priv_integrate_until(out, t, t_eval, event_options, std::forward<Callable>(observer), max_prints);
}

template<typename T, size_t N>
template<typename Callable>
OdeSolution<T, N> ODE<T, N>::rich_integrate(const T& interval, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    solver_->start_interpolation();
    OdeResult<T, N> res;
    this->integrate(&res, interval, event_options, std::forward<Callable>(observer), max_prints);
    OdeSolution<T, N> rich_res(std::move(res), *solver_->interpolator());
    solver_->stop_interpolation();
    return rich_res;
}

template<typename T, size_t N>
template<typename ArrayType, typename Callable>
bool ODE<T, N>::priv_integrate_until(OdeResult<T, N>* out, const T& t_max, const ArrayType& t_store, const std::vector<EventOptions>& event_options, Callable&& observer, int max_prints){
    if (solver_->is_dead()){
        if (out){
            *out = OdeResult<T, N>({}, {this->Nsys()}, solver_->diverges(), 0, false, 0, solver_->status());
        }
        return false;
    }else if (t_max*solver_->direction() < solver_->t()*solver_->direction()){
        if (out){
            *out = OdeResult<T, N>({}, {this->Nsys()}, 0, false, false, 0, "Cannot integrate in opposite direction");
        }
        return false; //cannot integrate in opposite direction
    }
    TimePoint TIME_START = Clock::now();
    constexpr bool store_explicit_steps = !std::is_same_v<std::decay_t<ArrayType>, EmptyArr<T>>;
    // ------------------------------ IMPLEMENTATION --------------------------------------
    solver_->resume();
    const T         t0 = solver_->t();
    const bool      first_eval_t0 = (t_store.size() > 0 && t_store[0] == t0);
    const char*     terminate_message = nullptr;
    const bool      include_first = (!store_explicit_steps || first_eval_t0);
    const size_t    t_start_idx = orbit_data_.t.size() - include_first;
    int             prints = 0;
    View1D<T>       t_eval(t_store.data() + first_eval_t0, t_store.size() - first_eval_t0);
    
    for (size_t i=0; i < event_data_.size(); i++){
        cached_idx_[i] = event_data_.data(i).size();
    }

    //check that all names in max_events are valid
    const std::vector<EventOptions> options = this->validate_events(event_options);
    EventCounter<T, N>              event_counter(options);

    auto event_state_valid = [&]()LAMBDA_INLINE{
        bool res = false;
        for (const size_t& ev : solver_->event_col()){
            if (event_counter.count_it(ev)){
                res = true;
                register_event(ev);
            }
        }
        return res;
    };

    // Since we pass an array of t_eval in the solver later, if step_ptr is not null,
    // it is guaranteed to point to an element in t_eval.
    auto main_observer = [&](const T& t, const T* q, const T* step_ptr) LAMBDA_INLINE -> bool {
        const bool at_valid_event = solver_->at_event() && event_state_valid();

        if constexpr (!store_explicit_steps) {
            // step_ptr is true only at the last step
            register_state();
        } else if (step_ptr) {
            register_state();
        }

        if (at_valid_event && !event_counter.is_running()) {
            terminate_message = "Max events reached";
            return false;
        }

        // =========================== Manage console output ==========================
        if (max_prints > 0){
            T percentage = (solver_->t() - t0)/(t_max-t0);
            if (percentage*max_prints >= prints){
                #pragma omp critical
                {
                    std::cout << std::setprecision(std::log10(max_prints)+1) << "\033[2K\rProgress: " << 100*percentage << "%" <<   "    Events: " << event_counter.total() << std::flush;
                    prints++;
                }

            }

        }
        // ============================================================================
        
        return observer(t, q, step_ptr);
    };

    bool success;
    if constexpr (store_explicit_steps){
        success = solver_->observe_until(t_max, main_observer, t_eval);
    }else{
        success = solver_->observe_until(t_max, main_observer);
    }

    if (success) {
        terminate_message = "t-goal";
    } else if (!terminate_message){
        terminate_message = solver_->status().c_str();
    }

    TimePoint       TIME_END = Clock::now();
    double          duration = Clock::as_duration(TIME_START, TIME_END);
    _runtime +=     duration;
    if (out){
        EventData<T>    event_res(this->event_data_, cached_idx_);
        OdeResult<T, N> res(orbit_data_, event_res, t_start_idx, solver_->diverges(), success, duration, terminate_message);
        *out = std::move(res);
    }
    return success;
}



template<typename T, size_t N>
bool ODE<T, N>::diverges() const{
    return solver_->diverges();
}

template<typename T, size_t N>
bool ODE<T, N>::is_dead() const{
    return solver_->is_dead();
}

template<typename T, size_t N>
size_t ODE<T, N>::n_points() const{
    return orbit_data_.t.size();
}

template<typename T, size_t N>
View1D<T> ODE<T, N>::t()const{
    return View1D<T>{orbit_data_.t.data(), this->n_points()};
}

template<typename T, size_t N>
View2D<T, 0, N> ODE<T, N>::q()const{
    return View2D<T, 0, N>{orbit_data_.q.data(), this->n_points(), this->Nsys()};
}

template<typename T, size_t N>
const T& ODE<T, N>::t(size_t i)const{
    assert(i < this->n_points() && "Index out of range");
    return orbit_data_.t[i];
}

template<typename T, size_t N>
View1D<T, N> ODE<T, N>::q(size_t i)const{
    return View1D<T, N>(orbit_data_.q.data() + i*this->Nsys(), this->Nsys());
}

template<typename T, size_t N>
const OrbitData<T>& ODE<T, N>::event_data(const std::string& event) const{
    return event_data_.data(event);
}

template<typename T, size_t N>
double ODE<T, N>::runtime()const{
    return _runtime;
}

template<typename T, size_t N>
const OdeRichSolver<T, N>* ODE<T, N>::solver()const{
    return solver_.ptr();
}

template<typename T, size_t N>
void ODE<T, N>::set_obj(const void* obj){
    solver_->set_obj(obj);
}

template<typename T, size_t N>
void ODE<T, N>::clear(){
    orbit_data_.clear_points();
    event_data_.clear_points();
    std::ranges::fill(cached_idx_, 0);
    register_state();
}

template<typename T, size_t N>
void ODE<T, N>::reset(){
    _runtime = 0;
    solver_->reset();
    this->clear();
}

template<typename T, size_t N>
template<typename RhsType, typename JacType>
void ODE<T, N>::init(MAIN_CONSTRUCTOR(T), EVENTS events, Integrator method){
    solver_.take_ownership(get_virtual_solver<T, N>(method, ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events).release());
    cached_idx_.resize(events.size(), 0);
    register_state();
    for (size_t i=0; i<events.size(); i++){
        event_data_.allocate_event(events[i]->name());
    }
}

template<typename T, size_t N>
void ODE<T, N>::register_state(){
    orbit_data_.add_point(solver_->t(), solver_->vector().data());
}

template<typename T, size_t N>
void ODE<T, N>::register_event(size_t i){
    event_data_.add_event(i, solver_->t(), solver_->vector().data());
}


template<typename T, size_t N>
std::vector<EventOptions> ODE<T, N>::validate_events(const std::vector<EventOptions>& options)const{


    size_t Nevs = solver_->event_col().size();
    std::vector<EventOptions> res(Nevs);
    bool found;
    for (size_t i=0; i<options.size(); i++) {
        found = false;
        for (size_t j=0; j<Nevs; j++){
            if (solver_->event_col().event(j).name() == options[i].name){
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
            if (option.name == solver_->event_col().event(i).name()){
                found = true;
                res[i] = option;
                res[i].max_events = ndspan::max(option.max_events, -1);
                break;
            }
        }
        if (!found){
            res[i] = {solver_->event_col().event(i).name()};
        }
    }
    return res;
}



} // namespace ode

#endif // ODE_INT_IMPL_HPP