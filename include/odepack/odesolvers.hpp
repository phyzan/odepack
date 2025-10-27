#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include "interpolators.hpp"
#include "events.hpp"

template<typename T, size_t N>
struct SolverState{
    
    State<T, N> state;
    std::vector<const Event<T, N>*> events;
    bool diverges;
    bool is_running;
    bool is_dead;
    size_t Nt;
    std::vector<std::string> event_names;
    std::string message;

    SolverState(const State<T, N>& state, const std::vector<const Event<T, N>*>& events, bool diverges, bool is_running, bool is_dead, size_t Nt, std::string message): state(state), events(events), diverges(diverges), is_running(is_running), is_dead(is_dead), Nt(Nt), event_names(events.size()), message(std::move(message)) {
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

        std::cout << "\n" << std::setprecision(precision) << 
        "OdeSolver current state:\n---------------------------\n"
        "\tt          : " << state.t << "\n" <<
        "\tq          : " << state.vector.repr(precision) << "\n" <<
        "\th          : " << state.habs << "\n" <<
        "\tDiverges   : " << (diverges ? "true" : "false") << "\n" << 
        "\tRunning    : " << (is_running ? "true" : "false") << "\n" <<
        "\tUpdates    : " << Nt << "\n" <<
        "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
        "\tState      : " << message << "\n" << 
        (event_names.size() == 0 ? "\tEvents     : No event\n" : "\tEvents     : " + event_message);
    }

};

template<typename T, size_t N>
class OdeSolver{

public:
    
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

    virtual ~OdeSolver() = default;

    // ACCESSORS

    virtual const T&                        t() const = 0;
    virtual const Array1D<T, N>&            q() const = 0;
    virtual const Array1D<T, N>&            true_vector() const = 0;
    virtual const T&                        stepsize() const = 0;
    virtual int                             direction() const = 0;
    virtual const T&                        rtol() const = 0;
    virtual const T&                        atol() const = 0;
    virtual const T&                        min_step() const = 0;
    virtual const T&                        max_step() const = 0;
    virtual const std::vector<T>&           args() const = 0;
    virtual size_t                          Nsys() const = 0;
    virtual size_t                          Nupdates() const = 0;
    virtual bool                            diverges() const = 0;
    virtual bool                            is_running() const = 0;
    virtual bool                            is_dead() const = 0;
    virtual std::string                     message() const = 0;
    virtual  SolverState<T, N>              state() const = 0;
    virtual std::vector<const Event<T, N>*> current_events() const = 0;
    virtual const EventCollection<T, N>&    event_col() const = 0;
    virtual std::string                     name() const = 0;
    virtual T                               auto_step(const ICS<T, N>* ics = nullptr) const = 0;
    virtual OdeSolver<T, N>*                clone() const = 0;
    virtual const Interpolator<T, N>*       interpolator() const = 0;
    virtual bool                            is_interpolating() const = 0;
    virtual bool                            at_event() const = 0;
    virtual const State<T, N>&              ics() const = 0;
    UniqueClone                             safe_clone() const;

    //MODIFIERS

    virtual bool                            advance()=0;
    virtual void                            stop(std::string text = "") = 0;
    virtual void                            kill(std::string text = "") = 0;
    virtual void                            set_tmax(T tmax) = 0;
    virtual bool                            resume() = 0;
    virtual void                            start_interpolation() = 0;
    virtual void                            stop_interpolation() = 0;
    virtual void                            reset() = 0;
    virtual void                            set_obj(const void* obj) = 0;

protected:

    OdeSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeSolver);
};

template<typename T, size_t N>
OdeSolver<T, N>::UniqueClone OdeSolver<T, N>::safe_clone() const{
    return std::unique_ptr<OdeSolver<T, N>>(this->clone());
}




#endif



