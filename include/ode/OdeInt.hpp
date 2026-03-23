#ifndef ODE_HPP
#define ODE_HPP

#include "SolverDispatcher.hpp"
#include "OdeResult/OdeResult.hpp"

namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================

struct EventOptions{
    std::string name;
    int max_events=-1;
    bool terminate=false;
    int period=1;
};


template<typename T, size_t N>
class EventCounter{

public:

    EventCounter(const std::vector<EventOptions>& options);

    DEFAULT_RULE_OF_FOUR(EventCounter);

    int operator[](size_t i) const;

    bool count_it(size_t i);

    bool is_running() const;

    bool can_fit(size_t event)const;

    size_t total()const;

private:

    std::vector<EventOptions> _options;
    std::vector<int> _counter;
    std::vector<int> _period_counter;
    size_t _total=0;
    bool _is_running = true;
};

template<typename T, size_t N=0>
class ODE{

public:

    template<typename RhsType, typename JacType>
    ODE(ODE_CONSTRUCTOR(T));

    DEFAULT_RULE_OF_FOUR(ODE)

    virtual ~ODE() = default;

    void                        Rhs(T* out, const T& t, const T* q) const;

    void                        Jac(T* out, const T& t, const T* q, const T* dt = nullptr) const;        

    virtual ODE<T, N>*          clone() const;

    std::unique_ptr<ODE<T, N>>  safe_clone() const;

    size_t                      Nsys() const;

    template<typename Callable = decltype(VoidFunc)>
    OdeSolution<T, N>           rich_integrate(const T& interval, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    template<typename Callable = decltype(VoidFunc)>
    bool                        integrate(OdeResult<T, N>* out, const T& interval, const std::vector<T>& t_array, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    template<typename Callable = decltype(VoidFunc)>
    bool                        integrate(OdeResult<T, N>* out, const T& interval, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    template<typename Callable = decltype(VoidFunc)>
    bool                        integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    template<typename Callable = decltype(VoidFunc)>
    bool                        integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<T>& t_eval, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    bool                        diverges() const;

    bool                        is_dead() const;

    size_t                      n_points() const;

    View1D<T>                   t() const;

    const T&                    t(size_t i) const;

    View2D<T, 0, N>             q() const;

    View1D<T, N>                q(size_t i) const;

    const OrbitData<T>&         event_data(const std::string& event) const;

    double                      runtime() const;

    const OdeRichSolver<T, N>*  solver() const;

    void                        set_obj(const void* obj);

    virtual void                clear();

    virtual void                reset();

protected:

    ODE(size_t nsys);

    PolyWrapper<OdeRichSolver<T, N>> solver_;
    OrbitData<T> orbit_data_;
    EventData<T> event_data_;
    std::vector<size_t> cached_idx_;
    double _runtime = 0;

    template<typename RhsType, typename JacType>
    void                                        init(ODE_CONSTRUCTOR(T));

    virtual void                                register_state();

    virtual void                                register_event(size_t i);

private:

    template<typename ArrayType, typename Callable = decltype(VoidFunc)>
    bool                                        priv_integrate_until(OdeResult<T, N>* out, const T& t_max, const ArrayType& t_store, const std::vector<EventOptions>& event_options={}, Callable&& observer = decltype(VoidFunc)(), int max_prints = 0);

    std::vector<EventOptions>                   validate_events(const std::vector<EventOptions>& options)const;

};

} // namespace ode

#endif
