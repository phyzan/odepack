#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include "interpolators.hpp"
#include "events.hpp"
#include <memory>

template<typename T, size_t N>
using VirtualInterp = std::unique_ptr<Interpolator<T, N>>;

struct EmptySolver{};

template<typename T, size_t N=0>
class OdeSolver{

public:
    
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

    virtual ~OdeSolver() = default;

    // ODE PROPERTIES
    virtual void                rhs(T* dq_dt, const T& t, const T* q) const = 0;
    virtual void                jac(T* jm, const T& t, const T* q) const = 0;

    // ACCESSORS
    virtual const T&            t() const = 0;
    virtual View1D<T, N>  vector() const = 0;
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
    virtual const std::string&  message() const = 0;
    virtual void                show_state(int prec=8) const = 0;
    virtual State<T>      state() const = 0;
    virtual State<T>      ics() const = 0;
    virtual bool                validate_ics(T t0, const T* q0) const = 0;
    virtual const std::string&  method() const = 0;
    virtual void                interp(T* result, const T& t) const = 0;
    virtual VirtualInterp<T, N> state_interpolator(int bdr1, int bdr2) const = 0;
    virtual T                   auto_step(T t, const T* q) const = 0;
    virtual T                   auto_step() const = 0;
    virtual OdeSolver<T, N>*    clone() const = 0;
    UniqueClone                 safe_clone() const{
        return std::make_unique<OdeSolver<T, N>>(this->clone());
    }

    // MODIFIERS
    virtual bool                advance() = 0;
    virtual bool                advance_until(T time) = 0;
    virtual void                reset() = 0;
    virtual bool                resume() = 0;
    virtual void                stop(const std::string& text = "") = 0;
    virtual void                kill(const std::string& text = "") = 0;
    virtual void                set_obj(const void* obj) = 0;
    virtual void                set_args(const T* new_args) = 0;

protected:

    OdeSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeSolver);

};


template<typename T, size_t N=0>
class OdeRichSolver : public OdeSolver<T, N>{

public:
    
    using UniqueClone = std::unique_ptr<OdeRichSolver<T, N>>;

    // ACCESSORS
    virtual View1D<T, N>                    true_vector() const = 0;
    virtual EventView<T>                    current_events() const = 0;
    virtual const EventCollection<T>&       event_col() const = 0;
    virtual const Interpolator<T, N>*       interpolator() const = 0;
    virtual bool                            is_interpolating() const = 0;
    virtual bool                            at_event() const = 0;
    virtual OdeRichSolver<T, N>*            clone() const = 0;
    UniqueClone                             safe_clone() const{
        return std::make_unique<OdeRichSolver<T, N>>(this->clone());
    }

    // MODIFIERS
    virtual bool                            advance_to_event()=0;

    virtual void                            set_tmax(T tmax) = 0;
    virtual void                            start_interpolation() = 0;
    virtual void                            stop_interpolation() = 0;

protected:

    OdeRichSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeRichSolver);
};


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

#endif