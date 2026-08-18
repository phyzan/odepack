#ifndef SOLVER_STATE_IMPL_HPP
#define SOLVER_STATE_IMPL_HPP

#include "SolverState.hpp"

namespace ode{
    
template<typename T, size_t N>
SolverState<T, N>::SolverState(const T* q, T t, T habs, size_t nsys, bool diverges, bool is_running, bool is_dead, size_t updates, std::string message)
: vector(q, nsys), msg(std::move(message)), nt(updates), time(t), stepsize(habs), diverging(diverges), running(is_running), dead(is_dead) {}

template<typename T, size_t N>
void SolverState<T, N>::show(int precision) const{
    std::cout << "\n" << std::setprecision(precision) << 
    "OdeSolver current state:\n---------------------------\n"
    "\ttime       : " << time << "\n" <<
    "\tq          : ";
    
    array_repr(std::cout, vector);
    std::cout << "\n" <<
    "\tstepsize   : " << stepsize << "\n" <<
    "\tDiverges   : " << (diverging ? "true" : "false") << "\n" << 
    "\tRunning    : " << (running ? "true" : "false") << "\n" <<
    "\tUpdates    : " << nt << "\n" <<
    "\tDead       : " << (dead ? "true" : "false") << "\n" <<
    "\tState      : " << msg << std::endl;
}


template<typename T, size_t N>
SolverRichState<T, N>::SolverRichState(const T* q, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, std::string message, std::string event) : SolverState<T, N>(q, t, habs, Nsys, diverges, is_running, is_dead, Nt, std::move(message)), event_name(std::move(event)) {
}

template<typename T, size_t N>
void SolverRichState<T, N>::show(int precision) const{

    SolverState<T, N>::show(precision);
    std::cout << "\tEvent     : " << event_name << "\n" << std::endl;
}


} // namespace ode

#endif // SOLVER_STATE_IMPL_HPP