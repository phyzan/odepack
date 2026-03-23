#ifndef RICH_BASE_IMPL_HPP
#define RICH_BASE_IMPL_HPP

#include "RichBase.hpp"

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const T* RichSolver<Derived, T, N, SP, RhsType, JacType>::true_state_ptr() const{
    const T* base_ptr = Base::true_state_ptr();
    int d = this->direction();
    T base_time = base_ptr[0]*d;
    if (const size_t* ev_idx = _events.begin()){
        const AnyEvent<T>& event = _events.event(*ev_idx);
        const T* derived_ptr = event->state()->get_exposed();
        T derived_time = derived_ptr[0]*d;
        if (base_time < derived_time){
            return base_ptr;
        }else{
            return derived_ptr;
        }
    }else{
        return base_ptr;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const T* RichSolver<Derived, T, N, SP, RhsType, JacType>::last_true_state_ptr() const{
    const T* base_ptr = Base::last_true_state_ptr();
    int d = this->direction();
    T base_time = base_ptr[0]*d;
    if (const size_t* ev_idx = _events.last_begin()){
        const AnyEvent<T>& event = _events.event(*ev_idx);
        const T* derived_ptr = event->state()->get_exposed();
        T derived_time = derived_ptr[0]*d;
        if (base_time > derived_time){
            return base_ptr;
        }else{
            return derived_ptr;
        }
    }else{
        return base_ptr;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
View1D<T, N> RichSolver<Derived, T, N, SP, RhsType, JacType>::true_vector() const{
    // The overriden true_state_ptr returns the the pointer to the **exposed** state, although the "true" in the name.
    // The "true" actually refers to the fact that the state might not point to the solver's adapted state, but could point to
    // any point between the old adapted state and the newly adapted state.
    // However in the case of masked events with hidden mask, even though the "true" state (in the sense that the BaseSolver uses it) is the intermediate event state, the **true** true state is the masked state, which is the one that the solver will actually continue from. So the true_vector() returns the view of the **true** true state, which is the actual (not exposed) state of the event if at an event, and is the same as the base solver state otherwise.
    const T* state_ptr;
    const T* base_ptr = Base::true_state_ptr();
    int d = this->direction();
    T base_time = base_ptr[0]*d;
    if (const size_t* ev_idx = _events.begin()){
        const AnyEvent<T>& event = _events.event(*ev_idx);
        const T* derived_ptr = event->state()->get_true();
        T derived_time = derived_ptr[0]*d;
        if (base_time < derived_time){
            state_ptr = base_ptr;
        }else{
            state_ptr = derived_ptr;
        }
    }else{
        state_ptr = base_ptr;
    }
    return View1D<T, N>(state_ptr + 2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
EventView<T> RichSolver<Derived, T, N, SP, RhsType, JacType>::current_events() const{
    if (const size_t* ev_idx = _events.begin()){
        const AnyEvent<T>& event = _events.event(*ev_idx);
        int d = this->direction();
        const T& base_time = Base::true_state_ptr()[0];
        const T& event_time = event->state()->t();
        if (base_time * d < event_time * d){
            return EventView<T>(nullptr, nullptr, 0);
        }else{
            return _events.event_view();
        }
    }else{
        return EventView<T>(nullptr, nullptr, 0);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const EventCollection<T>& RichSolver<Derived, T, N, SP, RhsType, JacType>::event_col() const{
    return _events;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
const Interpolator<T, N>& RichSolver<Derived, T, N, SP, RhsType, JacType>::interpolator() const{
    return _cli;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::is_interpolating() const{
    return _interp_data;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::at_event(int event) const{
    assert((event == -1 || (event >= 0 && static_cast<size_t>(event) < _events.size())) && "Invalid event index passed to RichSolver::at_event");
    if (const size_t* ev_idx = _events.begin()){
        const T& event_time = _events.event(*ev_idx)->state()->t();
        const bool at_event_time = (this->t() == event_time);
        if (at_event_time && event != -1){
            for (const size_t idx : _events){
                if (idx == size_t(event)){
                    return true;
                }
            }
        } else {
            return at_event_time;
        }
    }
    return false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
int RichSolver<Derived, T, N, SP, RhsType, JacType>::event_idx(const std::string& name) const{
    return _events.event_idx(name);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::show_state(int prec) const{
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->status(), this->current_events()).show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RichSolver<Derived, T, N, SP, RhsType, JacType>::RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events) : Base(ARGS), _events(events), _cli(t0, q0 ? q0 : Array1D<T, N>(nsys).data(), nsys){
    _events.setup(t0, this->args().data(), this->args().size(), this->Nsys(), this->direction());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::adv_impl(){

    if (this->requires_new_start()){
        this->re_adjust(this->true_vector().data());
    }

    if (this->equiv_states()){
        if (Base::adv_impl()){
            State<T> old_state = this->old_state();
            State<T> new_state = this->new_state();
            _events.detect_all_between(old_state, new_state, [this](T* out, const T& t){
                this->interp_impl(out, t);
            });
            if (_interp_data){
                std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(0, -1);
                if (const EventState<T>* ev = _events.canon_state()){
                    r->adjust_end(ev->t());
                }
                this->add_interpolant(std::move(r));
            }
        }else{
            return false;
        }
    }else if (!Base::is_at_new_state()){
        const T& base_time = Base::true_state_ptr()[0];
        int d = this->direction();
        bool event_waiting = false;
        if (const size_t* ev_idx = _events.begin()){
            const AnyEvent<T>& event = _events.event(*ev_idx);
            const T& event_time = event->state()->t();
            event_waiting = (base_time * d < event_time * d);
        }

        if (Base::adv_impl()){
            if (const size_t* idx_ptr = _events.begin()){
                const AnyEvent<T>& event = _events.event(*idx_ptr);
                const T& event_time = event->state()->t();
                if (!event_waiting && base_time * d >= event_time * d){
                    _events.next_result();
                }
            }else{
                _events.next_result();
            }
        }else{
            return false;
        }
    }else{
        _events.next_result();
    }

    if (_events.begin()){
        if (_interp_data && this->requires_new_start()){
            if (!_events.canon_event()->hides_mask()){
                auto r = std::unique_ptr<Interpolator<T, N>>(new LocalInterpolator<T, N>(this->t(), this->true_vector().data(), this->Nsys()));
                _cli.adjust_end(this->t());
                this->add_interpolant(std::move(r));
            }
        }
    }

    if (_interp_data){
        _cli.close_end();
    }

    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::advance_to_event(int event){
    assert((event == -1 || (event >= 0 && static_cast<size_t>(event) < _events.size())) && "Invalid event index passed to advance_to_event");
    if (_events.size() == 0){
        return false;
    }
    do{
        if (!this->advance()){
            return false;
        }
    }while (!this->at_event(event));
    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::advance_to_event(const T& tmax, int event){
    assert((event == -1 || (event >= 0 && static_cast<size_t>(event) < _events.size())) && "Invalid event index passed to advance_to_event");
    if (_events.size() == 0){
        return false;
    }
    bool success = false;
    Base::advance_until(tmax, [&](const T& t, const T* state, const T* extra)LAMBDA_INLINE{
        if (this->at_event(event)){
            success = true;
            return false; // stop advancing
        }
        return true; // continue advancing
    });
    return success;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::start_interpolation(){
    if (!_interp_data){
        _interp_data = true;

        if (this->equiv_states()){
            this->reset_interpolator();
        }
        else{
            int bdr1 = 1;

            if (const size_t* ev_idx = _events.begin()){
                int d = this->direction();
                const T& base_time = Base::true_state_ptr()[0];
                const T& t_event = _events.state(*ev_idx).get_true()[0];
                if ((base_time * d < t_event * d) && _events.canon_event() && (t_event == _events.canon_state()->t()) && _events.canon_event()->hides_mask()){
                    this->reset_interpolator();
                    bdr1 = -1;
                }
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::stop_interpolation(){
    this->reset_interpolator();
    _interp_data = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::reset_impl(){
    Base::reset_impl();
    _events.reset(this->direction());
    stop_interpolation();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){
    Base::re_adjust_impl(new_vector);
    if (_interp_data){
        _cli.adjust_end(this->t());
    }
}

// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void RichSolver<Derived, T, N, SP, RhsType, JacType>::set_args_impl(const T* new_args){
    Base::set_args_impl(new_args);
    _events.set_args(new_args, this->args().size());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant){
    if (_cli.last_interpolant().interval().is_point() && interpolant->interval().start_bdr() == 0){
        interpolant->close_start();
    }
    _cli.expand_by_owning(std::move(interpolant));
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::requires_new_start() const{
    return _events.canon_event() && (_events.canon_state()->t() == this->t());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::equiv_states() const{
    return this->t() == Base::t_new();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::reset_interpolator(){
    _cli = LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys());
}

// ============================================================================


} // namespace ode


#endif // RICH_BASE_IMPL_HPP