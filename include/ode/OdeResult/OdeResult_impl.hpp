#ifndef ODE_RESULT_IMPL_HPP
#define ODE_RESULT_IMPL_HPP

#include "OdeResult.hpp"

namespace ode{

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