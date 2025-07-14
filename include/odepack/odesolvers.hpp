#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <string>
#include <memory>
#include "events.hpp"
#include "interpolators.hpp"


template<typename T, int N>
class OdeSolver{


public:
    
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;
    using InterpolatorList = std::vector<const Interpolator<T, N>*>;

    virtual ~OdeSolver() = default;

    virtual OdeRhs<T, N>                    ode_rhs() const = 0;
    virtual const T&                        t() const = 0;
    virtual const vec<T, N>&                q() const = 0;
    virtual const vec<T, N>&                q_true() const = 0;
    virtual const T&                        stepsize() const = 0;
    virtual const T&                        tmax() const = 0;
    virtual const int&                      direction() const = 0;
    virtual const T&                        rtol() const = 0;
    virtual const T&                        atol() const = 0;
    virtual const T&                        min_step() const = 0;
    virtual const T&                        max_step() const = 0;
    virtual const std::vector<T>&           args() const = 0;
    virtual const size_t&                   Nsys() const = 0;
    virtual const bool&                     diverges() const = 0;
    virtual const bool&                     is_running() const = 0;
    virtual const bool&                     is_dead() const = 0;
    virtual const vec<T, N>&                error()const = 0;
    virtual const std::string&              message() const = 0;
    virtual const SolverState<T, N>         state() const = 0;
    virtual const EventCollection<T, N>&    events()const = 0;
    virtual bool                            at_event() const = 0;
    virtual std::string                     event_name() const = 0;
    virtual const Event<T, N>&              current_event() const = 0;
    virtual const int&                      current_event_index() const = 0;
    virtual const std::string&              name() const = 0;
    virtual T                               auto_step(T direction=0, const ICS<T, N>* ics = nullptr)const = 0;
    virtual OdeSolver<T, N>*                clone() const = 0;
    virtual UniqueClone                     safe_clone() const = 0;
    virtual UniqueClone                     with_new_events(const EventCollection<T, N>& events) const = 0;
    virtual InterpolatorList                interpolators() const = 0;

    virtual bool                            advance()=0;
    virtual bool                            set_goal(const T& t_max) = 0;
    virtual void                            stop(const std::string& text = "") = 0;
    virtual void                            kill(const std::string& text = "") = 0;
    virtual bool                            resume() = 0;
    virtual bool                            free() = 0;
    virtual void                            start_interpolation() = 0;
    virtual void                            stop_interpolation() = 0;
    virtual void                            clear_interpolators() = 0;

protected:

    OdeSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeSolver);
};




#endif



