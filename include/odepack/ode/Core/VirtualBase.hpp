#ifndef ODEPACK_VIRTUAL_BASE_HPP
#define ODEPACK_VIRTUAL_BASE_HPP

#include <functional>
#include "../Interpolation/Univariate/StateInterp.hpp"
#include "Events.hpp"
#include <polybox/polybox.hpp>
#include "VirtualTraits.hpp"

namespace ode {

using namespace interp::uni;

template<typename T, size_t N>
using VirtualInterp = std::unique_ptr<Interpolator<T, N>>;


template<typename T, size_t N, UtilPolicy UP>
using BoxedSolver = typename detail::SolverBoxSelector<T, N, UP>::type;

template<typename T, size_t N>
using BoxedInterp = pbox::Box<Interpolator<T, N>>;

template<typename T, size_t N=0>
class OdeSolver{

public:
    
    virtual ~OdeSolver() = default;

    // ODE PROPERTIES
    virtual void                get_rhs(T* dq_dt, const T& t, const T* q) const = 0;
    virtual void                get_jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const = 0;
    virtual void                get_jac_approx(T* j, const T& t, const T* q, const T* dt) const = 0;

    // ACCESSORS
    virtual const T&            get_time() const = 0;
    virtual const T&            get_new_time() const = 0;
    virtual const T&            get_old_time() const = 0;
    virtual View1D<T, N>        get_vector() const = 0;
    virtual View1D<T, N>        get_new_vector() const = 0;
    virtual View1D<T, N>        get_old_vector() const = 0;
    virtual State<T>            get_ics() const = 0;
    virtual State<T>            get_new_state() const = 0;
    virtual State<T>            get_old_state() const = 0;
    virtual const T&            get_stepsize() const = 0;
    virtual int                 get_direction() const = 0;
    virtual const T&            get_rtol() const = 0;
    virtual const T&            get_atol() const = 0;
    virtual const T&            get_min_step() const = 0;
    virtual const T&            get_max_step() const = 0;
    virtual size_t              get_nsys() const = 0;
    virtual size_t              get_step_count() const = 0;
    virtual bool                get_is_running() const = 0;
    virtual bool                get_is_dead() const = 0;
    virtual bool                get_diverges() const = 0;
    virtual const std::string&  get_status() const = 0;
    virtual bool                get_validate_ics(T t0, const T* q0) const = 0;
    virtual Integrator          get_method() const = 0;
    virtual void                get_interp(T* result, const T& t) const = 0;
    virtual size_t              get_rhs_eval_count() const = 0;
    virtual size_t              get_jac_eval_count() const = 0;
    virtual VirtualInterp<T, N> get_state_interpolator(int bdr1, int bdr2) const = 0;
    virtual T                   get_auto_step(T t, const T* q) const = 0;
    virtual T                   get_auto_step() const = 0;
    virtual void                show_state(int prec=8) const = 0;
    virtual std::unique_ptr<OdeSolver<T, N>> clone() const = 0;

    // MODIFIERS
    virtual bool                do_advance() = 0;
    virtual bool                do_advance_by(T interval) = 0;
    virtual bool                do_advance_until(const T& time) = 0;
    virtual bool                do_observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer) = 0;
    virtual bool                do_observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer, View1D<T> extra_steps) = 0;
    virtual BoxedInterp<T, N>   do_interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer = [](const auto&, const auto*, const auto*){return true;}) = 0;
    virtual void                do_reset() = 0;
    virtual bool                do_resume() = 0;
    virtual void                do_stop(const std::string& text = "") = 0;
    virtual void                do_kill(const std::string& text = "") = 0;
    virtual bool                do_set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0) = 0;

protected:

    OdeSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeSolver)

};


template<typename T, size_t N=0>
class OdeRichSolver : public OdeSolver<T, N>{

public:

    // ACCESSORS
    virtual const EventCollection<T>&       get_event_col() const = 0;
    virtual int                             get_event_idx(const std::string& name) const = 0;
    virtual bool                            get_at_event(int event_idx = -1) const = 0;
    virtual EventState<T>                   get_current_event() const = 0;
    virtual bool                            get_at_canon_event() const = 0;
    // MODIFIERS
    virtual bool                            do_advance_to_event(const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const std::vector<std::string>& event_names) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<std::string>& event_names) = 0;

protected:

    OdeRichSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeRichSolver)
};


template<Integrator M, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
struct SolverTypeGetter{
    using type = void;
};


} // namespace ode

#endif // ODEPACK_VIRTUAL_BASE_HPP