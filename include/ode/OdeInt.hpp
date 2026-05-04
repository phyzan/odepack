#ifndef ODE_HPP
#define ODE_HPP

#include "SolverDispatcher.hpp"
#include "OdeResult/OdeResult.hpp"

namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================


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

    template<hasRhsFunc<T> OdeType>
    ODE(ODE_CONSTRUCTOR(T));

    DEFAULT_RULE_OF_FOUR(ODE)

    virtual ~ODE() = default;

    void                        Rhs(T* out, const T& t, const T* q) const;

    void                        Jac(T* out, const T& t, const T* q, const T* dt = nullptr) const;        

    virtual ODE<T, N>*          clone() const;

    std::unique_ptr<ODE<T, N>>  safe_clone() const;

    size_t                      Nsys() const;

    template<OptionalObserver<T> Callable = std::nullptr_t>
    bool                        rich_integrate(OdeSolution<T, N>& out, const T& interval, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0);

    template<OptionalObserver<T> Callable = std::nullptr_t>
    bool                        integrate(OdeResult<T, N>* out, const T& interval, const std::vector<T>& t_array, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0);

    template<OptionalObserver<T> Callable = std::nullptr_t>
    bool                        integrate(OdeResult<T, N>* out, const T& interval, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0);

    template<OptionalObserver<T> Callable = std::nullptr_t>
    bool                        integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0);

    template<OptionalObserver<T> Callable = std::nullptr_t>
    bool                        integrate_until(OdeResult<T, N>* out, const T& t, const std::vector<T>& t_eval, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0);

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

    virtual void                clear();

    virtual void                reset();

protected:

    ODE(size_t nsys);

    pbox::owner<OdeRichSolver<T, N>> solver_;
    OrbitData<T> orbit_data_;
    EventData<T> event_data_;
    std::vector<size_t> cached_idx_;
    double _runtime = 0;

    template<hasRhsFunc<T> OdeType>
    void                                        init(ODE_CONSTRUCTOR(T));

    virtual void                                register_state();

    virtual void                                register_event(size_t i);

private:

    template<typename ArrayType, OptionalObserver<T> Callable = std::nullptr_t>
    bool                                        priv_integrate_until(OdeResult<T, N>* out, const T& t_max, const ArrayType& t_store, const std::vector<EventOptions>& event_options={}, Callable&& observer = nullptr, int max_prints = 0, bool interpolate = false);

};

} // namespace ode

#endif
