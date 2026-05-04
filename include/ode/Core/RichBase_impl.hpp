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
bool RichSolver<Derived, T, N, SP, OdeType>::adv_impl(Args&&... args){

    // When restarting from a masked event, then at the next step, the last state vector will display the masked state, whether the mask was hidden or not
    
    if (events.size() == 0){
        return Base::adv_impl(std::forward<Args>(args)...);
    } else if (this->at_canon_event()) {
        const MaskedState<T>* ms = events.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state. Report bug.");
        if (this->current_event().event->hides_mask()){
            Accessor::call_ReAdjust(*THIS, ms->masked_vector.data());
        } // if the mask is not hidden, the state has already been ReAdjusted
    }
    
    if (this->is_at_new_state()){
        if (!Base::adv_impl(std::forward<Args>(args)...)){
            // new event detection pass was triggered in this command
            return false;
        } else if (!this->push_event_queue()){
            is_at_event = false;
            is_at_canon_event = false; // is_event_waiting has been set to false in the previous adv_impl call, no need to set it again here
        }
        return true;
    }else if (is_event_waiting){
        if (Base::adv_impl(events.get_time(size_t(detection_idx+1)), std::forward<Args>(args)...)){
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
        return Base::adv_impl(std::forward<Args>(args)...);
    }
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(int event){
    assert((event == -1 || (event >= 0 && static_cast<size_t>(event) < events.size())) && "Invalid event index passed to advance_to_event");
    if (events.size() == 0){
        return false;
    }
    do{
        if (!this->advance()){
            return false;
        } else if (EventState<T> es = this->current_event()){
            if (event == -1 || es.idx == size_t(event)){
                return true;
            }
        }
    }while (true);
    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const T& tmax, int event){
    assert((event == -1 || (event >= 0 && static_cast<size_t>(event) < events.size())) && "Invalid event index passed to advance_to_event");
    if (events.size() == 0){
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
void RichSolver<Derived, T, N, SP, OdeType>::set_args_impl(const T* new_args){
    Base::set_args_impl(new_args);
    events.set_args(new_args, this->args().size());
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_canon_event() const{
    return is_at_canon_event;
}


// ============================================================================


} // namespace ode


#endif // RICH_BASE_IMPL_HPP