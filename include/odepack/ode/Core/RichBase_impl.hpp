#ifndef RICH_BASE_IMPL_HPP
#define RICH_BASE_IMPL_HPP

#include "RichBase.hpp"

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const EventCollection<T>& RichSolver<Derived, T, N, SP, OdeType>::event_col() const{
    return evt_col;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_event(int event_idx) const{
    if (event_idx == -1){
        return is_at_event;
    } else if (EventState<T> es = this->current_event()){
        return es.idx == size_t(event_idx);
    } else {
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
int RichSolver<Derived, T, N, SP, OdeType>::event_idx(const std::string& name) const{
    return evt_col.event_idx(name);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::vector<size_t> RichSolver<Derived, T, N, SP, OdeType>::toEventIdx(const std::vector<std::string>& event_names) const{
    std::vector<size_t> event_idx(event_names.size());
    for (size_t i = 0; i < event_names.size(); ++i){
        int idx = this->event_idx(event_names[i]);
        if (idx == -1){
            throw std::out_of_range("Invalid event name: " + event_names[i]);
        }
        event_idx[i] = size_t(idx);
    }
    return event_idx;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::show_state(int prec) const{
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->nsys(), this->diverges(), this->is_running(), this->is_dead(), this->step_count(), this->status(), this->current_event().event ? this->current_event().event->name() : "").show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
RichSolver<Derived, T, N, SP, OdeType>::RichSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> evs) : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir), evt_col(std::move(evs)) {
    this->evt_col.setup(t0, this->nsys(), this->direction());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool RichSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){

    // When restarting from a masked event, then at the next step, the last state vector will display the masked state, whether the mask was hidden or not
    
    if (evt_col.size() == 0){
        return Base::Adv_Impl(std::forward<Args>(args)...);
    } else if (this->at_canon_event()) {
        const MaskedState<T>* ms = evt_col.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state. Report bug.");
        if (this->current_event().event->hides_mask()){
            ODEPACK_CALL_DERIVED(ReAdjust, ms->masked_vector.data());
        } // if the mask is not hidden, the state has already been ReAdjusted
    }
    
    if (this->is_at_new_state()){
        if (!Base::Adv_Impl(std::forward<Args>(args)...)){
            // new event detection pass was triggered in this command
            return false;
        } else if (!this->push_event_queue()){
            is_at_event = false;
            is_at_canon_event = false; // is_event_waiting has been set to false in the previous Adv_Impl call, no need to set it again here
        }
        return true;
    }else if (is_event_waiting){
        if (Base::Adv_Impl(evt_col.get_time(size_t(detection_idx+1)), std::forward<Args>(args)...)){
            if (!this->push_event_queue()){
                is_at_event = false;
                is_at_canon_event = false;
            }
            return true;
        } else {
            return false;
        }
    }else{
        is_at_event = false;
        is_at_canon_event = false;
        return Base::Adv_Impl(std::forward<Args>(args)...);
    }
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const std::vector<size_t>& event_idx){
    for (size_t idx : event_idx){
        if (idx >= evt_col.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (evt_col.size() == 0){
        return false;
    }
    do{
        if (!this->advance()){
            return false;
        } else if (EventState<T> es = this->current_event()){
            if (event_idx.empty() || std::find(event_idx.begin(), event_idx.end(), es.idx) != event_idx.end()){
                return true;
            }
        }
    }while (true);
    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const T& tmax, const std::vector<size_t>& event_idx){
    for (size_t idx : event_idx){
        if (idx >= evt_col.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (evt_col.size() == 0){
        return false;
    }
    bool success = false;
    Base::advance_until(tmax, [&](const T& /*t*/, const T* /*state*/, const T* /*extra*/)NDSPAN_LAMBDA_INLINE{
        if (EventState<T> es = this->current_event()){
            if (event_idx.empty() || std::find(event_idx.begin(), event_idx.end(), es.idx) != event_idx.end()){
                success = true;
                return false; // stop advancing
            }
        }
        return true; // continue advancing
    });
    return success;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(event_idx);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const T& tmax, const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(tmax, event_idx);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::Reset(){
    Base::Reset();
    evt_col.reset(this->direction());
    current_idx = 0;
    detection_idx = -1;
    is_at_event = false;
    is_at_canon_event = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    is_at_canon_event = false;
    is_at_event = false;
    is_event_waiting = false;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::RequestTimeFloor(T& out) {
    // no need to call Base::RequestTimeFloor, the Base class does not request it.
    detection_idx = -1; // reset detection index at the start of a new detection round
    // do not also set is_at_event to false, as the Adv_Impl might fail and the step should remain in the same state.
    if ((is_event_waiting = evt_col.detect_all_between(this->old_state(), this->new_state(), [this](T* q_out, const T& t){
        this->interp_impl(q_out, t);
    }))){
        // is_event_waiting has been set to true, preparing the push_event_queue for the first event
        if (Base::RequestTimeFloor(out)){
            out = this->minimum_time(out, evt_col.get_time(0));
        } else {
            out = evt_col.get_time(0);
        }
        return true;
    } else {
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::push_event_queue(){
    if (is_event_waiting && evt_col.get_time(size_t(detection_idx+1)) == this->t()){
        current_idx = evt_col.get_event_idx(size_t(++detection_idx));
        // determine if this one is a canon event
        if (const MaskedState<T>* ms = evt_col.masked_state()){
            if ((is_at_canon_event = static_cast<bool>(ms->idx == current_idx)) && !evt_col.event(current_idx).hides_mask()){
                ODEPACK_CALL_DERIVED(ReAdjust, ms->masked_vector.data());
            }
        }
        // determine if there is another event after this one
        is_event_waiting = size_t(detection_idx) < evt_col.detection_size() - 1;
        is_at_event = true;
        return true;
    } else {
        return false;
    }
}

// PRIVATE METHODS


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_canon_event() const{
    return is_at_canon_event;
}


// ============================================================================


} // namespace ode


#endif // RICH_BASE_IMPL_HPP