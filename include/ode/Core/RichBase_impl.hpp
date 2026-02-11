#ifndef RICH_BASE_IMPL_HPP
#define RICH_BASE_IMPL_HPP

#include "RichBase.hpp"

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& RichSolver<Derived, T, N, SP, RhsType, JacType>::t_impl() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::t_impl();
        }else{
            return event.state()->t();
        }
    }else{
        return Base::t_impl();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* RichSolver<Derived, T, N, SP, RhsType, JacType>::vector_impl() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::vector_impl();
        }else{
            return _events.state(*ev_idx).exposed().vector();
        }
    }else{
        return Base::vector_impl();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 View1D<T, N> RichSolver<Derived, T, N, SP, RhsType, JacType>::true_vector() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return View1D<T, N>(Base::vector_impl(), this->Nsys());
        }else{
            return View1D<T, N>(_events.state(*ev_idx).true_state().vector(), this->Nsys());
        }
    }else{
        return View1D<T, N>(Base::vector_impl(), this->Nsys());
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
EventView<T> RichSolver<Derived, T, N, SP, RhsType, JacType>::current_events() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
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
const Interpolator<T, N>* RichSolver<Derived, T, N, SP, RhsType, JacType>::interpolator() const{
    return _cli.ptr();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::is_interpolating() const{
    return _interp_data;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
State<T> RichSolver<Derived, T, N, SP, RhsType, JacType>::state() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        if (Base::t_impl() * this->direction() < event.state()->t()*this->direction()){
            return Base::state();
        }else{
            return _events.state(*ev_idx).exposed();
        }
    }else{
        return Base::state();
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::at_event() const{
    if (const size_t* ev_idx = _events.begin()){
        const Event<T>& event = _events.event(*ev_idx);
        return static_cast<bool>(Base::t_impl() * this->direction() >= event.state()->t()*this->direction());
    }else{
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::show_state(int prec) const{
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message(), this->current_events()).show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
RichSolver<Derived, T, N, SP, RhsType, JacType>::RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events) : Base(ARGS), _events(events){
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
            _events.detect_all_between(old_state, new_state, interp_func<Derived, T, N, SP, RhsType, JacType>, this);
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

        bool event_waiting = false;
        if (const size_t* ev_idx = _events.begin()){
            const Event<T>& event = _events.event(*ev_idx);
            event_waiting = (Base::t_impl() * this->direction() < event.state()->t()*this->direction());
        }

        if (Base::adv_impl()){
            if (const size_t* idx_ptr = _events.begin()){
                const Event<T>& event = _events.event(*idx_ptr);
                if (!event_waiting && Base::t_impl() * this->direction() >= event.state()->t()*this->direction()){
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
                _cli->adjust_end(this->t());
                this->add_interpolant(std::move(r));
            }
        }
    }

    if (_interp_data){
        _cli->close_end();
    }

    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool RichSolver<Derived, T, N, SP, RhsType, JacType>::advance_to_event(){
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::start_interpolation(){
    if (!_interp_data){
        _interp_data = true;

        if (this->equiv_states()){
            _cli.take_ownership(new LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys()));
        }
        else{
            int bdr1 = 1;

            if (const size_t* ev_idx = _events.begin()){
                const T& t_event = _events.state(*ev_idx).t();
                if ((Base::t_impl() * this->direction() < t_event*this->direction()) && _events.canon_event() && (t_event == _events.canon_state()->t()) && _events.canon_event()->hides_mask()){
                    _cli.take_ownership(new LinkedInterpolator<T, N>(this->t(), this->vector().data(), this->Nsys()));
                    bdr1 = -1;
                }
            }


            std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(bdr1, -1);
            r->adjust_start(this->t());

            if (bdr1 == 1){
                _cli.take_ownership(new LinkedInterpolator<T, N>(r.get()));
            }
            else{
                _cli->expand_by_owning(std::move(r));
            }
        }

    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void RichSolver<Derived, T, N, SP, RhsType, JacType>::stop_interpolation(){
    _cli.take_ownership(nullptr);
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
        _cli->adjust_end(this->t());
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
    if (_cli->last_interpolant().interval().is_point() && interpolant->interval().start_bdr() == 0){
        interpolant->close_start();
    }
    _cli->expand_by_owning(std::move(interpolant));
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool RichSolver<Derived, T, N, SP, RhsType, JacType>::requires_new_start() const{
    return _events.canon_event() && (_events.canon_state()->t() == this->t());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool RichSolver<Derived, T, N, SP, RhsType, JacType>::equiv_states() const{
    return this->t_impl() == Base::t_new();
}

// ============================================================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void interp_func(T* res, const T& t, const void* obj){
    const auto* solver = reinterpret_cast<const RichSolver<Derived, T, N, SP, RhsType, JacType>*>(obj);
    solver->interp_impl(res, t);
}


} // namespace ode


#endif // RICH_BASE_IMPL_HPP