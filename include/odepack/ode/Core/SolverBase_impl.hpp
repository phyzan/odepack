#ifndef SOLVERBASE_IMPL_HPP
#define SOLVERBASE_IMPL_HPP

#include "SolverBase.hpp"
#include "../Tools_impl.hpp"
#include "FinDiff.hpp"
#include <odepack/ode/Tools.hpp>
#include <stdexcept>

#define NOW \
std::chrono::high_resolution_clock::now()

#define DURATION(T1, T2) std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(T2 - T1).count()

namespace ode{


// ODE PROPERTIES

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Rhs(T* dq_dt, const T& t, const T* q) const{
    return ode_.Rhs(dq_dt, t, q);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::rhs(T* dq_dt, const T& t, const T* q) const{
    this->Rhs(dq_dt, t, q);
    this->rhs_eval_count_++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Jac(T* jm, const T& t, const T* q, const T* dt) const{
    
    if constexpr (JP == JacPolicy::Approx){
        return this->jac_approx(jm, t, q, dt);
    } else if constexpr (JP == JacPolicy::Autodiff){
        // TODO : maybe use only the second branch for both cases
        // (for large N, the compiler might explode from the double template recursion)
        decltype(auto) scratch_duals = this->scratch_.duals();
        if constexpr (N > 0){
            NDSPAN_FOR_LOOP(I, N,
                scratch_duals[I+N] = DualType(q[I], {.axis=I});
            );
            ode_.Rhs(scratch_duals.data(), t, scratch_duals.data()+N);
            const DualType* rhs = scratch_duals.data();
            NDSPAN_FOR_LOOP(I, N,
                NDSPAN_FOR_LOOP(J, N,
                    jm[I + J*N] = rhs[I].get_diff_wrt(J);
                );
            );
            return;
        } else {
            const size_t nsys = this->nsys();
            const size_t nvars_default = DualType::get_default_nvars();
            DualType::set_default_nvars(nsys);
            for (size_t i=0; i<nsys; i++){
                scratch_duals[i + nsys] = DualType(q[i], {.axis=int(i)});
            }
            ode_.Rhs(scratch_duals.data(), t, scratch_duals.data() + nsys);
            const DualType* rhs = scratch_duals.data();
            for (size_t i=0; i<nsys; i++){
                for (size_t j=0; j<nsys; j++){
                    jm[i + j*nsys] = rhs[i].get_diff_wrt(j);
                }
            }
            DualType::set_default_nvars(nvars_default);
        }
    } else {
        ode_.Jac(jm, t, q);
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::jac(T* jm, const T& t, const T* q, const T* dt) const{
    this->Jac(jm, t, q, dt);
    this->jac_eval_count_++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::jac_approx(T* out, const T& t, const T* q, const T* dt) const{
    const size_t n = this->nsys();

    decltype(auto) scratch = this->scratch_.four_state_cache();

    ode::jac_approx<T>([this](T* dummy_out, const T& dummy_t, const T* dummy_q){
        this->Rhs(dummy_out, dummy_t, dummy_q);
    }, out, scratch.data(), t, q, dt, this->atol(), n);
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t() const{
    return this->true_state_ptr()[0];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector() const{
    return View1D<T, N>(this->true_state_ptr()+2, this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_new() const{
    return View1D<T, N>(this->new_state_ptr()+2, this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_old() const{
    return View1D<T, N>(this->old_state_ptr()+2, this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::stepsize() const{
    return new_state_[1];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
int BaseSolver<Derived, T, N, SP, OdeType>::direction() const{
    return direction_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::rtol() const{
    return rtol_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
 const T& BaseSolver<Derived, T, N, SP, OdeType>::atol() const{
    return atol_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::min_step() const{
    return min_step_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::max_step() const{
    return max_step_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::step_count() const{
    return step_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_running() const{
    return is_running_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_dead() const{
    return is_dead_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::diverges() const{
    return diverges_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const std::string& BaseSolver<Derived, T, N, SP, OdeType>::status() const{
    return msg_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::show_state(int prec) const{
    SolverState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->nsys(), this->diverges(), this->is_running(), this->is_dead(), this->step_count(), this->status()).show(prec);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::new_state() const{
    return State<T>(this->new_state_ptr(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::old_state() const{
    return State<T>(this->old_state_ptr(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::ics() const{
    return State<T>(this->ics_ptr(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp(T* out, const T& t) const{
    assert((t*this->direction() >= this->t_old()*this->direction() && t*this->direction() <= this->interp_new_state_ptr()[0]*this->direction()) && "Out of bounds interpolation requested");
    if (this->t_old() == this->t_new()){
        ndspan::copy_array(out, this->new_state_ptr(), this->nsys());
    }
    return interp_impl(out, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::rhs_eval_count() const{
    return rhs_eval_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::jac_eval_count() const{
    return jac_eval_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T BaseSolver<Derived, T, N, SP, OdeType>::auto_step(T t, const T* q) const{
    //returns absolute value of emperically determined first step.

    const int dir = this->direction();
    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    size_t n = this->nsys();
    T h0, d2, h1;

    decltype(auto) scratch = this->scratch_.four_state_cache();
    T* y1 = scratch.data();

    T* f1 = y1+n;
    T* scale = y1+2*n;
    T* f0 = y1+3*n;
    for (size_t i=0; i<n; i++){
        scale[i] = atol() + abs<T>(q[i])*rtol();
    }
    this->rhs(f0, t, q);
    T d0 = rms_norm(q, scale, n);
    T d1 = rms_norm(f0, scale, n);
    if (d0 * 100000 < 1 || d1 * 100000 < 1){
        h0 = T(1)/1000000;
    }
    else{
        h0 = d0/d1/100;
    }
    for (size_t i=0; i<n; i++){
        y1[i] = q[i]+h0*dir*f0[i];
    }
    this->rhs(f1, t+h0*dir, y1);
    T* tmp = y1; //y1 can be recycled, its not used anymore below
    for (size_t i=0; i<n; i++){
        tmp[i] = f1[i] - f0[i];
    }
    d2 = rms_norm(tmp, scale, n) / h0;
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = ndspan::max<T>(T(1)/1000000, h0/1000);
    }else{
        h1 = pow(100*ndspan::max<T>(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }
    return ndspan::max<T>(ndspan::min_of_pack<T>(T(100*h0), h1, this->max_step()), this->min_step());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T BaseSolver<Derived, T, N, SP, OdeType>::auto_step() const{
    return auto_step(this->t(), this->vector().data());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::unique_ptr<typename BaseSolver<Derived, T, N, SP, OdeType>::CloneType> BaseSolver<Derived, T, N, SP, OdeType>::clone() const {
    return std::make_unique<Derived>(*THIS);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }
    return Accessor::call_Adv_Impl(*THIS);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<OptionalObserver<T> Callable>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::interpolate_until(const T& time, const Callable& observer){
    pbox::Box<LinkedInterpolator<T, N>> interp = pbox::make_box<LinkedInterpolator<T, N>>(this->t_old(), this->vector_old().data(), this->nsys());
    bool current_state_is_new = false;
    if (!this->is_at_new_state()){
        interp->expand_by_owning(this->state_interpolator(0, -1));
    }else{
        current_state_is_new = true;
    }

    const T t_start = this->t();
    if (this->advance_until(time, [&](const T& t, const T* q, const T* t_ptr){
        bool obs_res;
        if constexpr (Observer<Callable, T>){
            obs_res = observer(t, q, t_ptr);
        } else{
            obs_res = true;
        }
        if (obs_res){
            if (this->is_at_new_state()){
                if (current_state_is_new){
                    interp->expand_by_owning(this->state_interpolator(0, -1));
                }
                interp->expand_by_owning(std::make_unique<LocalInterpolator<T, N>>(this->t(), this->vector().data(), this->nsys()));
                current_state_is_new = true;
            } else if (current_state_is_new) {
                interp->expand_by_owning(this->state_interpolator(0, -1));
                current_state_is_new = false;
            }
            return true;
        } else {
            return false;
        }

    })){
        if (t_start != interp->t_start()){
            interp->adjust_start(t_start);
        }
        if (time != interp->t_end()){
            interp->adjust_end(time);
        }
        interp->close_end();
        return interp;
    } else {
        return BoxedInterp<T, N>();
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer){
    return this->interpolate_until(time, observer);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_until(const T& time){
    return this->advance_until(time, nullptr);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<OptionalObserver<T> Callable, typename ArrayType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_until(const T& time, const Callable& observer, const ArrayType& extra_steps){

    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }

    int d = this->direction();
    if (time == this->t()){
        return false;
    } else if (time*d < this->t()*d) {
        throw std::runtime_error(GetStr("Cannot advance until time ", time, " because it is in the opposite direction of integration. Current time is ", this->t(), " and direction is ", d, "."));
    }

    constexpr bool explicit_steps = !std::is_same_v<std::decay_t<ArrayType>, EmptyArr<T>>;
    const bool has_extra_steps = explicit_steps && extra_steps.size() > 0;
    const T& t_dual = has_extra_steps ? extra_steps[extra_steps.size() - 1] : time;

    bool success;
    auto evolve = [&]() NDSPAN_LAMBDA_INLINE -> bool {
        bool res;
        while ((res = (this->is_running() && Accessor::call_Adv_Impl(*THIS, time))) && (time != this->t())){
            bool obs_res;
            if constexpr (Observer<Callable, T>){
                obs_res = observer(this->t(), this->true_state_ptr()+2, nullptr);
            } else{
                obs_res = true;
            }
            if (!obs_res){
                // the observer itself might have advanced the solver to the same target time, so its worth making this check.
                return this->t() * d >= time * d;
            }
        }

        if (res){
            const T* t_ptr = (!explicit_steps || (has_extra_steps && t_dual == time)) ? &t_dual : nullptr;
            if constexpr (Observer<Callable, T>){
                observer(this->t(), this->true_state_ptr()+2, t_ptr);
            }            
            return true;
        }else{
            return this->t() * d >= time * d;
        }
    };

    if (!has_extra_steps){
        return evolve();
    }else if (extra_steps[extra_steps.size()-1]*d > time*d){
        throw std::runtime_error(GetStr("Invalid extra steps: last extra step is ", extra_steps[extra_steps.size()-1], " but target time is ", time, ". Extra steps must be in the same direction and between the current time and the target time."));
    }else{
        auto validate_idx = [&](size_t idx) NDSPAN_LAMBDA_INLINE -> size_t{
            if (extra_steps[idx]*d <= this->t()*d){
                throw std::runtime_error(GetStr("Invalid extra step: ", extra_steps[idx], ". Extra steps must be in the same direction and between the current time (", this->t(), ") and the target time (", time, ")."));
            }
            return idx;
        };
        size_t idx = 0;
        while (idx < extra_steps.size() && (success = (this->is_running() && this->advance_until(extra_steps[validate_idx(idx)], observer))) && (time != this->t())){
            idx++;
        }

        if (this->t() != time && success){
            return evolve();
        } else{
            return success;
        }
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_by(T interval){
    assert(interval >= 0 && "Interval must be non-negative in advance_by. Its sign is determined by the solver's direction of integration.");
    return this->advance_until(this->t() + interval*this->direction());
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer, View1D<T> extra_steps){
    return this->advance_until(time, observer, extra_steps);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::observe_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer){
    return this->advance_until(time, observer);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename Setter>
auto BaseSolver<Derived, T, N, SP, OdeType>::apply_ics_setter(T t0, Setter&& func, T stepsize){
    T* ics = const_cast<T*>(this->ics_ptr());
    return priv_apply_ics_setter(ics, t0, std::forward<Setter>(func), stepsize);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename Setter>
auto BaseSolver<Derived, T, N, SP, OdeType>::restart_from_modified_state(T t0, Setter&& func, T stepsize){
    T* ics = const_cast<T*>(this->ics_ptr());
    ndspan::copy_array(ics+2, this->vector().data(), this->nsys());
    return priv_apply_ics_setter(ics, t0, std::forward<Setter>(func), stepsize);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::set_ics(T t0, const T* y0, T stepsize, int direction){

    assert((direction == 1 || direction == -1 || direction == 0) && "Direction must be 1, -1, or 0");
    direction = (direction == 0) ? this->direction() : direction; // if 0, keep existing direction;
    if (this->validate_ics(t0, y0)){
        if (stepsize < 0) {
            this->cerr("Cannot set negative stepsize in solver initialization");
            return false;
        } else if (stepsize == 0) {
            this->direction_ = direction;
            stepsize = this->auto_step(t0, y0);
        }else{
            this->direction_ = direction;
        }

        T* ics = const_cast<T*>(this->ics_ptr());
        ics[0] = t0;
        ics[1] = stepsize;
        ndspan::copy_array(ics+2, y0, this->nsys());
        THIS->Reset();
        return true;
    }else {
#ifndef DPK_NO_WARN
        this->cerr("Tried to set invalid initial conditions");
#endif
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::stop(const std::string& text){
    if (!this->is_running()){
        return;
    }
    this->is_running_ = false;
    this->set_message((text == "") ? "Stopped by user" : text);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::kill(const std::string& text){
    if (this->is_dead()){
        return;
    }
    this->is_running_ = false;
    this->is_dead_ = true;
    this->msg_ = (text == "") ? "Killed by user" : text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::resume(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else{
        this->set_message("Running");
        this->is_running_ = true;
        return true;
    }
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::unique_ptr<Interpolator<T, N>> BaseSolver<Derived, T, N, SP, OdeType>::state_interpolator(int bdr1, int bdr2) const{
    auto interp = this->local_interp();
    const T* s1 = this->old_state_ptr();
    const T* s2 = this->interp_new_state_ptr();
    return std::make_unique<CustomLocalInterpolator<T, N, decltype(interp)>>(std::move(interp), s1[0], s2[0], s1+2, s2+2, this->nsys(), bdr1, bdr2);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
StepResult BaseSolver<Derived, T, N, SP, OdeType>::adapt_impl(T* state, const T* old_state){
    return ODEPACK_CALL_DERIVED(adapt_impl, state, old_state);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp_impl(T* result, const T& t) const{
    ODEPACK_CALL_DERIVED(interp_impl, result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
auto BaseSolver<Derived, T, N, SP, OdeType>::local_interp() const{
    return Accessor::call_local_interp(*THIS);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Reset(){
    this->msg_ = "Running";
    this->step_count_ = 0;
    this->rhs_eval_count_ = 0;
    this->jac_eval_count_ = 0;
    this->use_new_state_ = true;
    this->is_running_ = true;
    this->is_dead_ = false;
    this->diverges_ = false;
    old_state_ = ics_state_;
    new_state_ = ics_state_;
    true_state_ = ics_state_;
    interp_state_ = ics_state_;
}

//=============================================================================

// OVERRIDEN IN RICH SOLVER


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::true_state_ptr() const{
    return true_state_.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool BaseSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){
    const int d = this->direction();
    if constexpr (sizeof...(Args) > 0){
        T time_floor = minimum_time(args...);
        if (this->is_at_new_state() && (time_floor*d <= t_new()*d)){
            return false;
        } else if (this->is_at_new_state()){
            decltype(auto) scratch_state = this->scratch_.state();
            StepResult result = this->adapt_impl(scratch_state.data(), this->new_state_ptr());
            if (validate_it(result, scratch_state.data())){
                // ======== update internal states ========
                std::swap(old_state_, new_state_);
                std::swap(new_state_, scratch_state);
                for (size_t i=0; i<scratch_state.size(); i++){
                    true_state_[i] = new_state_[i];
                }
                use_new_state_ = true;
                step_count_++;
                // ==========================================
                T new_floor;
                if (ODEPACK_CALL_DERIVED(RequestTimeFloor, new_floor)){
                    assert((new_floor*d > t_old()*d && new_floor*d <= t_new()*d) && "Invalid floor requested, with additional requests");
                    time_floor = minimum_time(new_floor, time_floor);
                }
                
                if (time_floor*d < t_new()*d){
                    this->move_state(time_floor);
                }
                return true;
            }else{
                return false;
            }
        } else if (time_floor*d < t_new()*d){
            this->move_state(time_floor);
            return true;
        } else {
            this->move_state(t_new());
            return true;
        }
    } else if (this->is_at_new_state()){
        decltype(auto) scratch_state = this->scratch_.state();
        StepResult result = this->adapt_impl(scratch_state.data(), this->new_state_ptr());
        if (validate_it(result, scratch_state.data())){
            // ======== update internal states ========
                std::swap(old_state_, new_state_);
                std::swap(new_state_, scratch_state);
                for (size_t i=0; i<scratch_state.size(); i++){
                    true_state_[i] = new_state_[i];
                }
                use_new_state_ = true;
                step_count_++;
            // ==========================================
            T new_floor;
            if (ODEPACK_CALL_DERIVED(RequestTimeFloor, new_floor) && new_floor*d < t_new()*d){
                assert((new_floor*d > t_old()*d && new_floor*d <= t_new()*d) && "Invalid floor requested without additional requests.");
                this->move_state(new_floor);
            }
            return true;
        } else {
            return false;
        }
    } else {
        this->move_state(t_new());
        return true;
    }

}

// HELPER METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::ics_ptr() const{
    return this->ics_state_.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::new_state_ptr() const{
    return this->new_state_.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::old_state_ptr() const{
    return this->old_state_.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::interp_new_state_ptr() const{
    if (this->use_new_state_){
        return this->new_state_ptr();
    }else{
        return this->interp_state_.data(); // 5th index reserved for interpolation purposes
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_new() const{
    return this->new_state_[0];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_old() const{
    return this->old_state_[0];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_message(const std::string& text){
    this->msg_ = text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::warn_paused() const{
#ifndef DPK_NO_WARN
    this->cerr("\nSolver has paused integrating. Resume before advancing.");
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::warn_dead() const{
#ifndef DPK_NO_WARN
    this->cerr("\nSolver has permanently stopped integrating. Termination cause:\n\t" + this->msg_);
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::ReAdjust(const T* new_vector){
    ndspan::copy_array(this->interp_state_.data(), this->new_state_ptr(), this->nsys()+2); //store the re-adjusted new state for interpolation
    T* state = true_state_.data();
    state[0] = this->t();
    state[1] = this->stepsize();
    ndspan::copy_array(state+2, new_vector, this->nsys());
    if (! is_at_new_state_){
        new_state_ = true_state_;
    }
    use_new_state_ = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics(T t0, const T* q0) const {
    return ODEPACK_CALL_DERIVED(validate_ics_impl, t0, q0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics_impl(T t0, const T* q0) const {

    if (!all_are_finite(q0, this->nsys()) || !isfinite(t0)){
        return false;
    }

    decltype(auto) scratch_state = this->scratch_.ics_cache();

    /*
    Calling "this", not "THIS". Derived classes that override Rhs can have their version validated.
    However since this function might be called before the Derived classes has been fully constructed,
    calling "THIS" could lead to undefined behavior.
    */
    this->Rhs(scratch_state.data(), t0, q0);

    return all_are_finite(scratch_state.data(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_at_new_state() const{
    return is_at_new_state_;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
MutView<T, Layout::F, N, N> BaseSolver<Derived, T, N, SP, OdeType>::jac_view(T* j) const{
    //returns a high level view of the jacobian matrix, so that its elements
    //can be accessed using matrix(i, j). This function simply simplifies
    //the process of constructing the correct object that can safely view the jacobian matrix
    //by doing
    // auto matrix = solver->jac_view(jac_ptr);
    // matrix(i, j) = ...
    return MutView<T, Layout::F, N, N>(j, this->nsys(), this->nsys());
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BaseSolver<Derived, T, N, SP, OdeType>::BaseSolver(SOLVER_CONSTRUCTOR(T)) : 
    ics_state_(q0.size()+2),
    old_state_(q0.size()+2),
    new_state_(q0.size()+2),
    true_state_(q0.size()+2),
    interp_state_(q0.size()+2),
    rtol_(rtol),
    atol_(atol),
    min_step_(min_step),
    max_step_(max_step),
    scratch_(q0.size()),
    ode_(std::move(ode)),
    nsys_(q0.size()),
    direction_(dir){
        assert(this->nsys() > 0 && "Ode system size is 0");
        if (stepsize < 0){
            throw std::runtime_error("The stepsize argument cannot be negative");
        }
        if (max_step < min_step){
            throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
        }
        
        if (q0.data() == nullptr){
            this->kill("Initial conditions not set (nullptr provided)");
        } else if (this->validate_ics_impl(t0, q0.data())){
            T habs = (stepsize == 0 ? this->auto_step(t0, q0.data()) : abs<T>(stepsize));
            ics_state_[0] = t0;
            ics_state_[1] = habs;
            ndspan::copy_array(ics_state_.data()+2, q0.data(), this->nsys());
            old_state_ = ics_state_;
            new_state_ = ics_state_;
            true_state_ = ics_state_;
            interp_state_ = ics_state_;
        } else {
            this->kill("Initial conditions contain nan or inf, or ode(ics) does");
        }
}


// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_it(StepResult result, const T* state){
    bool success = true;
    switch (result){
        case StepResult::Success:
            break;
        case StepResult::INF_ERROR:
            this->set_message("ODE solution diverges (inf or nan encountered)");
            this->diverges_ = true;
            success = false;
            break;
        case StepResult::TINY_STEP_ERROR:
            this->kill("Required stepsize was smaller than machine precision");
            success = false;
            break;
        case StepResult::MIN_STEP_ERROR:
            this->kill("The next time step is smaller than the minimum allowed step");
            success = false;
            break;
        case StepResult::MAX_STEP_ERROR:
            this->kill("The next time step is larger than the maximum allowed step");
            success = false;
            break;
    }
    if (success && (state[0] == this->t_new())){
        this->kill("The next time step is identical to the previous one, possibly due to machine rounding error");
        success = false;
    }

    if (!success){
        //close the interpolation interval as most integration algorithms
        //alter their interpolation polynomials when calling adapt_impl,
        //but since the step failed, the current interpolation interval is no longer valid.
        use_new_state_ = false;;
        ndspan::copy_array(interp_state_.data(), this->old_state_ptr(), this->nsys()+2);
    }

    return success;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::move_state(const T& time){
    assert( (time*direction() <= this->t_new()*direction()) && "Out of bounds time requested in move_state");

    if (time != this->t_new()) {
        set_state(time, true_state_.data());
        is_at_new_state_ = false;
    }else if (! is_at_new_state()){
        // update the true state to the new state, because time is exactly at t_new
        is_at_new_state_ = true;
        true_state_ = new_state_;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_state(const T& time, T* state){
    state[0] = time;
    state[1] = this->stepsize();
    interp(state+2, time);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename Setter>
auto BaseSolver<Derived, T, N, SP, OdeType>::priv_apply_ics_setter(T* ics, T t0, Setter&& func, T stepsize){
    ics[0] = t0;
    if constexpr (std::is_void_v<std::invoke_result_t<Setter, T*>>){
        func(ics+2);
        assert(all_are_finite(ics+2, this->nsys()) && "Invalid ics in apply_ics_setter");
        if (stepsize < 0) {
            this->cerr("Cannot set negative stepsize in solver initialization");
        } else if (stepsize == 0) {
            stepsize = this->auto_step(t0, ics+2);
        }
        ics[1] = stepsize;
        THIS->Reset();
    } else {
        auto res = func(ics+2);
        assert(all_are_finite(ics+2, this->nsys()) && "Invalid ics in apply_ics_setter");
        if (stepsize < 0) {
            this->cerr("Cannot set negative stepsize in solver initialization");
        } else if (stepsize == 0) {
            stepsize = this->auto_step(t0, ics+2);
        }
        ics[1] = stepsize;
        THIS->Reset();
        return res;
    }
}


} // namespace ode

#endif // SOLVERBASE_IMPL_HPP