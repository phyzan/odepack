#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <functional>
#include "../Interpolation/Univariate/StateInterp.hpp"
#include "Events.hpp"
#include "Dispatcher.hpp"

namespace ode {

template<typename T, size_t N>
using VirtualInterp = std::unique_ptr<Interpolator<T, N>>;

struct EmptySolver{};

template<typename T, size_t N=0>
class OdeSolver{

public:
    
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

    virtual ~OdeSolver() = default;

    // ODE PROPERTIES
    virtual void                Rhs(T* dq_dt, const T& t, const T* q) const = 0;
    virtual void                Jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const = 0;
    virtual void                jac_approx(T* j, const T& t, const T* q, const T* dt) const = 0;

    // ACCESSORS
    virtual const T&            t() const = 0;
    virtual const T&            t_last() const = 0;
    virtual const T&            t_new() const = 0;
    virtual const T&            t_old() const = 0;
    virtual View1D<T, N>        vector() const = 0;
    virtual View1D<T, N>        vector_last() const = 0;
    virtual View1D<T, N>        vector_new() const = 0;
    virtual View1D<T, N>        vector_old() const = 0;
    virtual State<T>            ics() const = 0;
    virtual State<T>            state() const = 0;
    virtual State<T>            last_state() const = 0;
    virtual State<T>            new_state() const = 0;
    virtual State<T>            old_state() const = 0;
    virtual const T&            stepsize() const = 0;
    virtual int                 direction() const = 0;
    virtual const T&            rtol() const = 0;
    virtual const T&            atol() const = 0;
    virtual const T&            min_step() const = 0;
    virtual const T&            max_step() const = 0;
    virtual const Array1D<T>&   args() const = 0;
    virtual size_t              Nsys() const = 0;
    virtual size_t              Nupdates() const = 0;
    virtual bool                is_running() const = 0;
    virtual bool                is_dead() const = 0;
    virtual bool                diverges() const = 0;
    virtual const std::string&  status() const = 0;
    virtual void                show_state(int prec=8) const = 0;
    virtual bool                validate_ics(T t0, const T* q0) const = 0;
    virtual Integrator          method() const = 0;
    virtual void                interp(T* result, const T& t) const = 0;
    virtual size_t              n_evals_rhs() const = 0;
    virtual size_t              n_evals_jac() const = 0;
    virtual VirtualInterp<T, N> state_interpolator(int bdr1, int bdr2) const = 0;
    virtual T                   auto_step(T t, const T* q) const = 0;
    virtual T                   auto_step() const = 0;
    virtual OdeSolver<T, N>*    clone() const = 0;

    // MODIFIERS
    virtual bool                advance() = 0;
    virtual bool                advance_by(T interval) = 0;
    virtual bool                advance_until(const T& time) = 0;
    virtual bool                observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer) = 0;
    virtual bool                observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer, View1D<T> extra_steps) = 0;
    virtual pbox::Box<Interpolator<T, N>>  interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer = [](const auto&, const auto*, const auto*){return true;}) = 0;
    virtual void                Reset() = 0;
    virtual bool                resume() = 0;
    virtual void                stop(const std::string& text = "") = 0;
    virtual void                kill(const std::string& text = "") = 0;
    virtual void                set_args(const T* new_args) = 0;
    virtual bool                set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0) = 0;

protected:

    OdeSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeSolver);

};


template<typename T, size_t N=0>
class OdeRichSolver : public OdeSolver<T, N>{

public:
    
    using UniqueClone = std::unique_ptr<OdeRichSolver<T, N>>;

    // ACCESSORS
    virtual const EventCollection<T>&       event_col() const = 0;
    virtual int                             event_idx(const std::string& name) const = 0;
    virtual bool                            at_event() const = 0;
    virtual EventState<T>                   current_event() const = 0;
    virtual OdeRichSolver<T, N>*            clone() const = 0;

    // MODIFIERS
    virtual bool                            advance_to_event(int event = -1) = 0;
    virtual bool                            advance_to_event(const T& tmax, int event = -1) = 0;

protected:

    OdeRichSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeRichSolver);
};

enum class UtilPolicy : std::uint8_t{ Virtual, RichVirtual};

enum class SolverPolicy : std::uint8_t{ Static, RichStatic, Virtual, RichVirtual};

template<typename T, size_t N, SolverPolicy SP>
struct HelperVirtualSolver{ using type = EmptySolver;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::Virtual>{ using type = OdeSolver<T, N>;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::RichVirtual>{ using type = OdeRichSolver<T, N>;};

template<typename T, size_t N, SolverPolicy SP>
using BaseInterface = typename HelperVirtualSolver<T, N, SP>::type;


template<typename Solver, typename T, size_t N, SolverPolicy SP>
using SolverCloneType = std::conditional_t<SP==SolverPolicy::Virtual, OdeSolver<T, N>, std::conditional_t<SP==SolverPolicy::RichVirtual, OdeRichSolver<T, N>, Solver>>;

template<SolverPolicy SP>
constexpr bool is_rich = (SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual);

/// @brief Traits struct mapping a solver type to its virtual base interface.
/// Specialize this before the solver class definition to override the default.
template<typename Derived, typename T, size_t N, SolverPolicy SP>
struct SolverVirtualTypeTraits {
    using type = BaseInterface<T, N, SP>;
};



} // namespace ode

#endif