#ifndef RICH_SOLVER_HPP
#define RICH_SOLVER_HPP

#include "solverbase.hpp"

#define EVENTS const std::vector<const Event<T, N>*>&



template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void interp_func(T* res, const T& t, const void* obj);

template<typename Derived, typename T, size_t N, SolverPolicy SP>
class RichSolver : public BaseSolver<Derived, T, N, SP>{

    using Base = BaseSolver<Derived, T, N, SP>;

    friend Base; // So that Base can access specific private methods for static override
    friend void interp_func<Derived, T, N, SP>(T*, const T&, const void*);

public:

    // ACCESSORS
    inline View1D<const T, N>               true_vector() const;
    std::vector<const Event<T, N>*>         current_events() const;
    const EventCollection<T, N>&            event_col() const;
    const Interpolator<T, N>*               interpolator() const;
    bool                                    is_interpolating() const;
    State<const T>                          state() const;
    bool                                    at_event() const;
    void                                    show_state(int prec=8) const;

    // MODIFIERS
    bool                                    advance_until(T time);
    bool                                    advance_to_event();
    void                                    set_tmax(T tmax);
    void                                    start_interpolation();
    void                                    stop_interpolation();

    RichSolver() = delete;

protected:

    RichSolver(SOLVER_CONSTRUCTOR(T, N), std::vector<const Event<T, N>*> events) : Base(ARGS), _events(include_tmax_event(events)), _cli(t0, q0, nsys){
        this->initialize_events(t0);
    }
    
    DEFAULT_RULE_OF_FOUR(RichSolver)
    ~RichSolver() = default;

    void reset_impl();

private:

    //================= STATIC OVERRIDES ======================
    inline const T&                         t_impl() const;
    inline View1D<const T, N>               vector_impl() const;
    bool                                    adv_impl();
    inline void             set_args_impl(const T* new_args);
    //=========================================================


    void                    initialize_events(const T& t0);
    void                    add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant);
    inline bool             requires_new_start() const;
    inline bool             equiv_states() const;

    inline static std::vector<const Event<T, N>*>& include_tmax_event(std::vector<const Event<T, N>*>& events){
        static TmaxEvent<T, N> ev;
        events.insert(events.begin(), &ev);
        return events;
    }

    EventCollection<T, N>       _events;
    LinkedInterpolator<T, N>    _cli;
    long int                    _event_idx = -1;
    bool                        _interp_data = false;

};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& RichSolver<Derived, T, N, SP>::t_impl() const{
    return (_event_idx == -1) ? Base::t_impl() : _events.state(_event_idx).t();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<const T, N> RichSolver<Derived, T, N, SP>::vector_impl() const{
    switch (_event_idx) {
        case -1:
            return Base::vector_impl();
        default:
            const T* vec = _events.state(_event_idx).exposed().vector();
            return View1D<const T, N>(vec, this->Nsys());
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<const T, N> RichSolver<Derived, T, N, SP>::true_vector() const{
    switch (_event_idx) {
        case -1:
            return Base::vector_impl();
        default:
            const T* vec = _events.state(_event_idx).True().vector();
            return View1D<const T, N>(vec, this->Nsys());
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
std::vector<const Event<T, N>*> RichSolver<Derived, T, N, SP>::current_events() const{
    std::vector<const Event<T, N>*> events(0);
    for (const size_t* i=_events.begin(); i != _events.end(); ++i){
        events.push_back(&_events.event(*i));
    }
    events.shrink_to_fit();
    return events;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
const EventCollection<T, N>& RichSolver<Derived, T, N, SP>::event_col() const{
    return _events;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
const Interpolator<T, N>* RichSolver<Derived, T, N, SP>::interpolator() const{
    return &_cli;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool RichSolver<Derived, T, N, SP>::is_interpolating() const{
    return _interp_data;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
State<const T> RichSolver<Derived, T, N, SP>::state() const{
    switch (_event_idx) {
        case -1:
            return Base::state();
        default:
            return _events.state(_event_idx).exposed();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool RichSolver<Derived, T, N, SP>::at_event() const{
    return _event_idx != -1;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::show_state(int prec) const{
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message(), this->current_events()).show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool RichSolver<Derived, T, N, SP>::adv_impl(){

    if (this->requires_new_start()){
        this->remake_new_state(this->true_vector().data());
    }


    if (this->equiv_states()){
        if (Base::adv_impl()){
            State<const T> last_state = {this->last_true_state_ptr(), this->Nsys()};
            State<const T> true_state = {this->true_state_ptr(), this->Nsys()};
            _events.detect_all_between(last_state, true_state, interp_func<Derived, T, N, SP>, this);
            if (_interp_data){
                std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(0, -1);
                if (const EventState<T, N>* ev = _events.canon_state()){
                    r->adjust_end(ev->t());
                }
                this->add_interpolant(std::move(r));
            }
        }
        else{
            return false;
        }
    }
    else{
        _events.next_result();
    }


    if (_events.begin()){
        _event_idx = *_events.begin();
        if (_interp_data && this->requires_new_start()){
            if (!_events.canon_event()->hides_mask()){
                auto r = std::unique_ptr<Interpolator<T, N>>(new LocalInterpolator<T, N>(this->t(), this->true_vector().data(), this->Nsys()));
                _cli.adjust_end(this->t());
                this->add_interpolant(std::move(r));
            }
        }
    }else{
        _event_idx = -1;
    }
    for (size_t idx : _events){
        if (_events.event(idx).is_leathal()){
            this->kill(_events.event(idx).name());
        }
        else if (_events.event(idx).is_stop_event()){
            this->stop(_events.event(idx).name());
        }
    }

    if (_interp_data){
        // _current_linked_interpolator.adjust_end(this->t());
        _cli.close_end();
    }

    return true;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool RichSolver<Derived, T, N, SP>::advance_until(T time){
    this->set_tmax(time);
    while (this->is_running()) {
        this->advance();
    }

    assert(this->t()*this->direction() <= time*this->direction() && "Internal solver bug");
    if (this->t() == time){
        this->resume();
        return true;
    }else {
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool RichSolver<Derived, T, N, SP>::advance_to_event(){
    if (_events.size() == 0){
        return false;
    }
    do {
        if (!this->advance()){
            return false;
        }
    }while (!this->at_event());

    return true;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::set_tmax(T tmax){
    _events.set_tmax(tmax);
    if (tmax != this->t()){
        this->resume();
    }
    else{
        this->stop("t-goal");
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::start_interpolation(){
    if (!_interp_data){
        _interp_data = true;

        if (this->equiv_states()){
            _cli = LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys());
        }
        else{
            int bdr1 = 1;
            if (at_event() && _events.canon_event() && (_events.state(_event_idx).t() == _events.canon_state()->t()) && _events.canon_event()->hides_mask()){
                _cli = LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys());
                bdr1 = -1;
            }
            std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(bdr1, -1);
            r->adjust_start(this->t());

            if (bdr1 == 1){
                _cli = LinkedInterpolator<T, N>(r.get());
            }
            else{
                _cli.expand_by_owning(std::move(r));
            }

        }

    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::stop_interpolation(){
    _cli = LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys());
    _interp_data = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::reset_impl(){
    Base::reset_impl();
    _events.reset();
    _event_idx = -1;
    stop_interpolation();
}

// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void RichSolver<Derived, T, N, SP>::set_args_impl(const T* new_args){
    Base::set_args_impl(new_args);
    _events.set_args(new_args, this->args().size());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::initialize_events(const T& t0){
    _events.set_args(this->args().data(), this->args().size());
    _events.set_start(t0, this->direction());
    _events.set_array_size(this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void RichSolver<Derived, T, N, SP>::add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant){
    LinkedInterpolator<T, N>& cli = this->_cli;
    if (cli.last_interpolant().interval().is_point() && interpolant->interval().start_bdr() == 0){
        interpolant->close_start();
    }
    cli.expand_by_owning(std::move(interpolant));
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool RichSolver<Derived, T, N, SP>::requires_new_start() const{
    return _events.canon_event() && (_events.canon_state()->t() == this->t());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool RichSolver<Derived, T, N, SP>::equiv_states() const{
    return at_event() ? _events.state(_event_idx).t() == Base::t_impl() : true;
}

// ============================================================================

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void interp_func(T* res, const T& t, const void* obj){
    const auto* solver = reinterpret_cast<const RichSolver<Derived, T, N, SP>*>(obj);
    solver->interp_impl(res, t);
}



template<typename Derived, typename T, size_t N, SolverPolicy SP>
using BaseDispatcher = std::conditional_t<(SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual), RichSolver<Derived, T, N, SP>, BaseSolver<Derived, T, N, SP>>;

template<SolverPolicy SP>
constexpr bool is_rich = (SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual);

#endif