#ifndef SOLVERSTATE_HPP
#define SOLVERSTATE_HPP

#include "Core/Events.hpp"
#include "../ndspan/arrays.hpp"

namespace ode {

template<typename T, size_t N>
struct SolverState{
    
    Array1D<T> vector;
    T t;
    T habs;
    bool diverges;
    bool is_running;
    bool is_dead;
    size_t Nt;
    std::string message;

    SolverState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, std::string message);

    void show(const int& precision = 15) const;

};


template<typename T, size_t N>
struct SolverRichState : public SolverState<T, N>{

    std::vector<std::string> event_names;

    SolverRichState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, const std::string& message, const EventView<T>& events);

    void show(const int& precision = 15) const;

};

} // namespace ode

#endif