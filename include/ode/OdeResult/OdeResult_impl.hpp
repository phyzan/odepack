#ifndef ODE_RESULT_IMPL_HPP
#define ODE_RESULT_IMPL_HPP

#include "OdeResult.hpp"

namespace ode{

//===========================================================================================
//                                      OrbitData<T>
//===========================================================================================

template<typename T>
size_t OrbitData<T>::size() const {
    return t.size();
}

template<typename T>
View1D<T> OrbitData<T>::t_view() const {
    return View1D<T>(t.data(), t.size());
}

template<typename T>
View2D<T, 0, 0> OrbitData<T>::q_view() const {
    return View2D<T, 0, 0>(q.data(), t.size(), nsys);
}

template<typename T>
const T& OrbitData<T>::get_q(size_t i, size_t j) const {
    assert(i < t.size() && j < nsys && "Index out of range");
    return q[i * nsys + j];
}

template<typename T>
void OrbitData<T>::clear_points() {
    t = std::vector<T>();
    q = std::vector<T>();
}

template<typename T>
void OrbitData<T>::add_point(const T& t_val, const T* q_val) {
    t.push_back(t_val);
    try {
        q.insert(q.end(), q_val, q_val + nsys);
    } catch (...) {
        t.pop_back();  // rollback
        throw;
    }
}


//===========================================================================================
//                                      EventData<T>
//===========================================================================================

template<typename T>
EventData<T>::EventData(const EventData& other, const std::vector<size_t>& start_indices) : event_names_(other.event_names_), nsys_(other.nsys_) {
    assert(start_indices.size() == other.size() && "Start indices size must match the number of events");
    for (size_t i = 0; i < other.size(); ++i) {
        size_t start_idx = start_indices[i];
        assert(start_idx <= other.data(i).t.size() && "Start index must be within the bounds of the provided data");
        OrbitData<T> data;
        const OrbitData<T>& other_data = other.data(i);
        data.t = std::vector<T>(other_data.t.begin() + start_idx, other_data.t.end());
        data.q = std::vector<T>(other_data.q.begin() + start_idx * nsys_, other_data.q.end());
        data.nsys = other_data.nsys;
        event_data_.push_back(std::move(data));
    }
}

template<typename T>
size_t EventData<T>::size() const {
    return event_data_.size();
}

template<typename T>
size_t EventData<T>::nsys() const {
    return nsys_;
}

template<typename T>
const OrbitData<T>& EventData<T>::data(size_t i) const {
    assert(i < event_data_.size() && "Event index out of range");
    return event_data_[i];
}

template<typename T>
const OrbitData<T>& EventData<T>::data(const std::string& name) const {
    auto it = std::ranges::find(event_names_, name);
    if (it == event_names_.end()){
        throw std::runtime_error("Event name not found: " + name);
    }
    size_t index = std::distance(event_names_.begin(), it);
    return this->data(index);
}

template<typename T>
const std::string& EventData<T>::name(size_t i) const {
    assert(i < event_names_.size() && "Event index out of range");
    return event_names_[i];
}

template<typename T>
void EventData<T>::allocate_event(const std::string& name) {
    if (name.empty()){
        throw std::runtime_error("Event name cannot be empty");
    }
    if (std::ranges::find(event_names_, name) != event_names_.end()){
        throw std::runtime_error("Event name already exists");
    }
    event_data_.emplace_back(OrbitData<T>{.t = {}, .q = {}, .nsys = nsys_});

    try {
        event_names_.push_back(name);
    } catch (...) {
        event_data_.pop_back();  // rollback
        throw;
    }
}

template<typename T>
void EventData<T>::add_event(size_t event_idx, const T& t, const T* q) {
    if (event_idx >= event_names_.size()){
        throw std::runtime_error("Event index out of range");
    }
    OrbitData<T>& data = event_data_[event_idx];
    data.t.push_back(t);
    try {
        data.q.insert(data.q.end(), q, q + nsys_);
    } catch (...) {
        data.t.pop_back();  // rollback
        throw;
    }
}

template<typename T>
void EventData<T>::add_event(const std::string& name, const T& t, const T* q) {
    auto it = std::ranges::find(event_names_, name);
    if (it == event_names_.end()){
        throw std::runtime_error("Event name not found: " + name);
    }
    size_t index = std::distance(event_names_.begin(), it);
    this->add_event(index, t, q);
}

template<typename T>
void EventData<T>::clear_points() {
    for (auto& data : event_data_){
        data = OrbitData<T>{.t = {}, .q = {}, .nsys = nsys_};
    }
}


//===========================================================================================
//                                      OdeResult<T, N>
//===========================================================================================

template<typename T, size_t N>
OdeResult<T, N>::OdeResult(const OrbitData<T>& orbit_data, EventData<T> event_data, size_t orb_idx_start, bool diverges, bool success, double runtime, std::string message) : event_data_(std::move(event_data)), diverges_(diverges), success_(success), runtime_(runtime), message_(std::move(message)) {
    assert(orb_idx_start <= orbit_data.t.size() && "Start index must be within the bounds of the provided data");
    // Copy only the relevant portion of the data based on the provided start indices
    orbit_data_.t = std::vector<T>(orbit_data.t.begin() + orb_idx_start, orbit_data.t.end());
    orbit_data_.q = std::vector<T>(orbit_data.q.begin() + orb_idx_start * orbit_data.nsys, orbit_data.q.end());
    orbit_data_.nsys = orbit_data.nsys;
}

template<typename T, size_t N>
OdeResult<T, N>* OdeResult<T, N>::clone() const {
    return new OdeResult<T, N>(*this);
}

template<typename T, size_t N>
View1D<T> OdeResult<T, N>::t() const {
    return View1D<T>{orbit_data_.t.data(), orbit_data_.t.size()};
}

template<typename T, size_t N>
View2D<T, 0, N> OdeResult<T, N>::q() const {
    return View2D<T, 0, N>{orbit_data_.q.data(), orbit_data_.t.size(), orbit_data_.nsys};
}

template<typename T, size_t N>
const T& OdeResult<T, N>::q(size_t i, size_t j) const{
    assert(i < orbit_data_.t.size() && j < orbit_data_.nsys && "Index out of range");
    return orbit_data_.q[i * orbit_data_.nsys + j];
}

template<typename T, size_t N>
const EventData<T>& OdeResult<T, N>::event_data() const{
    return event_data_;
}

template<typename T, size_t N>
bool OdeResult<T, N>::diverges() const {
    return diverges_;
}

template<typename T, size_t N>
bool OdeResult<T, N>::success() const {
    return success_;
}

template<typename T, size_t N>
double OdeResult<T, N>::runtime() const {
    return runtime_;
}

template<typename T, size_t N>
const std::string& OdeResult<T, N>::message() const {
    return message_;
}

template<typename T, size_t N>
void OdeResult<T, N>::examine() const {
    std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n"
              << "\tPoints           : " << orbit_data_.t.size() << "\n"
              << "\tDiverges         : " << (diverges_ ? "true" : "false") << "\n"
              << "\tSuccess          : " << (success_ ? "true" : "false") << "\n"
              << "\tRuntime          : " << runtime_ << "\n"
              << "\tTermination cause: " << message_ << "\n"
              << event_log();
}

template<typename T, size_t N>
std::string OdeResult<T, N>::event_log() const {
    std::string res;
    res += "\tEvents:\n\t----------\n";
    for (size_t i=0; i<event_data_.size(); i++){
        const OrbitData<T>& data = event_data_.data(i);
        const std::string& name = event_data_.name(i);
        res += "\t    " + name + " : " + std::to_string(data.t.size()) + "\n";
    }
    res += "\n\t----------\n";
    return res;
}



// ==================================================================================
// ================================ OdeSolution =====================================
// ==================================================================================


template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(OrbitData<T> orbit_data, EventData<T> event_data, size_t orb_idx_start, bool diverges, bool success, double runtime, std::string message, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(std::move(orbit_data), std::move(event_data), orb_idx_start, diverges, success, runtime, std::move(message)), interpolator_(interpolator.clone()) {}

template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(std::move(other)), interpolator_(interpolator.clone()) {}

template<typename T, size_t N>
Array1D<T, N> OdeSolution<T, N>::operator()(const T& t) const{
    return interpolator_->make_call(t);
}

template<typename T, size_t N>
OdeSolution<T, N>* OdeSolution<T, N>::clone() const{ return new OdeSolution<T, N>(*this);}




}

#endif // ODE_RESULT_IMPL_HPP
