#ifndef ODE_RESULT_IMPL_HPP
#define ODE_RESULT_IMPL_HPP

#include "OdeResult.hpp"

namespace ode{

//===========================================================================================
//                                      OdeResult<T, N>
//===========================================================================================

template<typename T, size_t N>
OdeResult<T, N>::OdeResult(const std::vector<T>& t, const Array2D<T, 0, N>& q, EventMap event_map, bool diverges, bool success, double runtime, std::string message) : _t(t), _q(q), _event_map(std::move(event_map)), _diverges(diverges), _success(success), _runtime(runtime), _message(std::move(message)) {}



template<typename T, size_t N>
OdeResult<T, N>* OdeResult<T, N>::clone() const {
    return new OdeResult<T, N>(*this);
}

template<typename T, size_t N>
const std::vector<T>& OdeResult<T, N>::t() const {
    return _t;
}

template<typename T, size_t N>
const Array2D<T, 0, N>& OdeResult<T, N>::q() const {
    return _q;
}

template<typename T, size_t N>
template<std::integral INT1, std::integral INT2>
const T& OdeResult<T, N>::q(INT1 i, INT2 j) const{
    return _q(i, j);
}

template<typename T, size_t N>
const EventMap& OdeResult<T, N>::event_map() const {
    return _event_map;
}

template<typename T, size_t N>
bool OdeResult<T, N>::diverges() const {
    return _diverges;
}

template<typename T, size_t N>
bool OdeResult<T, N>::success() const {
    return _success;
}

template<typename T, size_t N>
double OdeResult<T, N>::runtime() const {
    return _runtime;
}

template<typename T, size_t N>
const std::string& OdeResult<T, N>::message() const {
    return _message;
}

template<typename T, size_t N>
void OdeResult<T, N>::examine() const {
    std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n"
              << "\tPoints           : " << _t.size() << "\n"
              << "\tDiverges         : " << (_diverges ? "true" : "false") << "\n"
              << "\tSuccess          : " << (_success ? "true" : "false") << "\n"
              << "\tRuntime          : " << _runtime << "\n"
              << "\tTermination cause: " << _message << "\n"
              << event_log();
}

template<typename T, size_t N>
std::string OdeResult<T, N>::event_log() const {
    std::string res;
    res += "\tEvents:\n\t----------\n";
    for (const auto& [name, array] : _event_map) {
        res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
    }
    res += "\n\t----------\n";
    return res;
}

template<typename T, size_t N>
std::vector<T> OdeResult<T, N>::t_filtered(const std::string& event) const {
    return _t_event_data(this->_t.data(), this->_event_map, event);
}

template<typename T, size_t N>
Array2D<T, 0, N> OdeResult<T, N>::q_filtered(const std::string& event) const {
    return _q_event_data<T, N>(this->_q.data(), this->_event_map, event, _q.Ncols());
}



// ==================================================================================
// ================================ OdeSolution =====================================
// ==================================================================================


template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(const std::vector<T>& t, const Array2D<T, 0, N>& q, const EventMap& event_map, bool diverges, bool success, double runtime, std::string message, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(t, q, event_map, diverges, success, runtime, message), _interpolator(interpolator.clone()) {}

template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(const OdeSolution& other) : OdeResult<T, N>(other), _interpolator(other._interpolator->clone()) {}

template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(OdeSolution&& other) noexcept : OdeResult<T, N>(std::move(other)), _interpolator(other._interpolator){
    other._interpolator = nullptr;
}

template<typename T, size_t N>
OdeSolution<T, N>::OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(std::move(other)), _interpolator(interpolator.clone()) {}

template<typename T, size_t N>
OdeSolution<T, N>& OdeSolution<T, N>::operator=(const OdeSolution& other){
    if (&other != this){
        delete _interpolator;
        _interpolator = other._interpolator->clone();
    }
    return *this;
}

template<typename T, size_t N>
OdeSolution<T, N>& OdeSolution<T, N>::operator=(OdeSolution&& other) noexcept{
    if (&other != this){
        delete _interpolator;
        _interpolator = other._interpolator;
        other._interpolator = nullptr;
    }
    return *this;
}

template<typename T, size_t N>
OdeSolution<T, N>::~OdeSolution(){
    delete _interpolator;
    _interpolator = nullptr;
}

template<typename T, size_t N>
Array1D<T, N> OdeSolution<T, N>::operator()(const T& t) const{
    return _interpolator->make_call(t);
}

template<typename T, size_t N>
OdeSolution<T, N>* OdeSolution<T, N>::clone() const{ return new OdeSolution<T, N>(*this);}




}

#endif // ODE_RESULT_IMPL_HPP