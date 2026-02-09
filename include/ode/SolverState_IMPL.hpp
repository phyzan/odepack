#ifndef SOLVER_STATE_IMPL_HPP
#define SOLVER_STATE_IMPL_HPP

#include "SolverState.hpp"

namespace ode{
    
template<typename T, size_t N>
SolverState<T, N>::SolverState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, std::string message): vector(vector, Nsys), t(t), habs(habs), diverges(diverges), is_running(is_running), is_dead(is_dead), Nt(Nt), message(std::move(message)) {}

template<typename T, size_t N>
void SolverState<T, N>::show(const int& precision) const{
    std::cout << "\n" << std::setprecision(precision) << 
    "OdeSolver current state:\n---------------------------\n"
    "\tt          : " << t << "\n" <<
    "\tq          : " << array_repr<T>(vector, precision) << "\n" <<
    "\th          : " << habs << "\n" <<
    "\tDiverges   : " << (diverges ? "true" : "false") << "\n" << 
    "\tRunning    : " << (is_running ? "true" : "false") << "\n" <<
    "\tUpdates    : " << Nt << "\n" <<
    "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
    "\tState      : " << message << std::endl;
}


template<typename T, size_t N>
SolverRichState<T, N>::SolverRichState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, const std::string& message, const EventView<T>& events) : SolverState<T, N>(vector, t, habs, Nsys, diverges, is_running, is_dead, Nt, message), event_names(events.size()) {
    for (size_t i=0; i<events.size(); i++){
        event_names[i] = events[i]->name();
    }
}

template<typename T, size_t N>
void SolverRichState<T, N>::show(const int& precision) const{
    std::string event_message;
    if (event_names.size() == 0){
        event_message = "";
    }
    else{
        for (size_t i=0; i<event_names.size()-1; i++){
            event_message += event_names[i] + ", ";
        }
        event_message += event_names[event_names.size()-1];
    }

    SolverState<T, N>::show(precision);
    std::cout << "\tEvents     : " + event_message + "\n" << std::endl;
}


} // namespace ode

#endif // SOLVER_STATE_IMPL_HPP