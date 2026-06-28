#ifndef RICH_BASE_IMPL_HPP
#define RICH_BASE_IMPL_HPP

#include "RichBase.hpp"

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const EventCollection<T>& RichSolver<Derived, T, N, SP, OdeType>::event_col() const{
    return events;
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
    return events.event_idx(name);
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
    SolverRichState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->status(), this->current_event().event).show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
RichSolver<Derived, T, N, SP, OdeType>::RichSolver(SOLVER_CONSTRUCTOR(T), std::vector<const Event<T>*> events) : Base(ARGS), events(events){
    this->events.setup(t0, this->args().data(), this->args().size(), this->Nsys(), this->direction());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool RichSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){

    // When restarting from a masked event, then at the next step, the last state vector will display the masked state, whether the mask was hidden or not
    
    if (events.size() == 0){
        return Base::Adv_Impl(std::forward<Args>(args)...);
    } else if (this->at_canon_event()) {
        const MaskedState<T>* ms = events.masked_state();
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
        if (Base::Adv_Impl(events.get_time(size_t(detection_idx+1)), std::forward<Args>(args)...)){
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
        if (idx >= events.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (events.size() == 0){
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
        if (idx >= events.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (events.size() == 0){
        return false;
    }
    bool success = false;
    Base::advance_until(tmax, [&](const T& t, const T* state, const T* extra)LAMBDA_INLINE{
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
    events.reset(this->direction());
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

// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::SetArgs(const T* new_args){
    Base::SetArgs(new_args);
    events.set_args(new_args, this->args().size());
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_canon_event() const{
    return is_at_canon_event;
}


// ============================================================================


} // namespace ode


#endif // RICH_BASE_IMPL_HPP