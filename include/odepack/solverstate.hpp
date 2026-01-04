#ifndef SOLVERSTATE_HPP
#define SOLVERSTATE_HPP

#include "events.hpp"
#include "../ndspan/arrays.hpp"

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

    SolverState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, std::string message): vector(vector, Nsys), t(t), habs(habs), diverges(diverges), is_running(is_running), is_dead(is_dead), Nt(Nt), message(std::move(message)) {}

    void show(const int& precision = 15) const{
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

};


template<typename T, size_t N>
struct SolverRichState : public SolverState<T, N>{

    std::vector<const Event<T, N>*> events;
    std::vector<std::string> event_names;

    SolverRichState(const T* vector, T t, T habs, size_t Nsys, bool diverges, bool is_running, bool is_dead, size_t Nt, const std::string& message, const std::vector<const Event<T, N>*>& events) : SolverState<T, N>(vector, t, habs, Nsys, diverges, is_running, is_dead, Nt, message), events(events), event_names(events.size()) {
        for (size_t i=0; i<events.size(); i++){
            event_names[i] = events[i]->name();
        }
    }

    void show(const int& precision = 15) const{
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
        std::cout << (event_names.size() == 0 ? "\tEvents     : No event\n" : "\tEvents     : " + event_message + "\n") << std::endl;
    }

};

#endif