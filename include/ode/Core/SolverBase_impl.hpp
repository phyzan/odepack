#ifndef SOLVERBASE_IMPL_HPP
#define SOLVERBASE_IMPL_HPP

#include "SolverBase.hpp"
#include "../Tools_impl.hpp"
#include "../Interpolation/Univariate/StateInterp_impl.hpp"
#include "Events_impl.hpp"
#include "FinDiff.hpp"

#define NOW \
std::chrono::high_resolution_clock::now()

#define DURATION(T1, T2) std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(T2 - T1).count()

namespace ode{


// ODE PROPERTIES

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Rhs(T* dq_dt, const T& t, const T* q) const{
    return _ode.Rhs(dq_dt, t, q, _args.data());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::rhs(T* dq_dt, const T& t, const T* q) const{
    this->Rhs(dq_dt, t, q);
    this->_n_evals_rhs++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Jac(T* jm, const T& t, const T* q, const T* dt) const{
    
    if constexpr (JP == JacPolicy::Approx){
        return this->jac_approx(jm, t, q, dt);
    } else if constexpr (JP == JacPolicy::Autodiff){
        FOR_LOOP(size_t, I, N,
            _diff_worker[I+N] = DualType(q[I], autodiff::Variable<I>{});
        );
        _ode.Rhs(_diff_worker.data(), DualType(t), _diff_worker.data()+N, _args_worker.data());

        const DualType* rhs = _diff_worker.data();
        FOR_LOOP(size_t, I, N,
            FOR_LOOP(size_t, J, N,
                jm[I + J*N] = rhs[I].diff_value(J);
            );
        );
        return;
    } else {
        _ode.Jac(jm, t, q, _args.data());
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::jac(T* jm, const T& t, const T* q, const T* dt) const{
    THIS->Jac(jm, t, q, dt);
    this->_n_evals_jac++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::jac_approx(T* out, const T& t, const T* q, const T* dt) const{
    const size_t n = this->Nsys();

    // ----------- Used only if N > 0 --------------
    static thread_local std::array<T, 4*N> work;
    // ---------------------------------------------

    T* worker;

    if constexpr (N > 0){
        worker = work.data();
    } else{
        if (_cache_4.size() != 4*n){
            _cache_4.resize(4, n);
        }
        worker = _cache_4.data();
    }

    ode::jac_approx<T>([this](T* out, const T& t, const T* q){
        this->Rhs(out, t, q);
    }, out, worker, t, q, dt, this->atol(), n);
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t() const{
    return this->true_state_ptr()[0];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector() const{
    return View1D<T, N>(this->true_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_last() const{
    return View1D<T, N>(this->last_true_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_new() const{
    return View1D<T, N>(this->new_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_old() const{
    return View1D<T, N>(this->old_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::stepsize() const{
    return this->_state_data(_new_state_idx, 1);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
int BaseSolver<Derived, T, N, SP, OdeType>::direction() const{
    return _direction;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::rtol() const{
    return this->_scalar_data[rtol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
 const T& BaseSolver<Derived, T, N, SP, OdeType>::atol() const{
    return this->_scalar_data[atol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::min_step() const{
    return this->_scalar_data[min_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::max_step() const{
    return this->_scalar_data[max_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const Array1D<T>& BaseSolver<Derived, T, N, SP, OdeType>::args() const{
    return _args;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::Nupdates() const{
    return _Nupdates;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_running() const{
    return _is_running;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_dead() const{
    return _is_dead;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::diverges() const{
    return _diverges;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const std::string& BaseSolver<Derived, T, N, SP, OdeType>::status() const{
    return _message;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::show_state(int prec) const{
    SolverState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->status()).show(prec);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::new_state() const{
    return State<T>(this->new_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::old_state() const{
    return State<T>(this->old_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::state() const{
    return State<T>(this->true_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::last_state() const{
    return State<T>(this->last_true_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
State<T> BaseSolver<Derived, T, N, SP, OdeType>::ics() const{
    return State<T>(this->ics_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
Integrator BaseSolver<Derived, T, N, SP, OdeType>::method() const{
    return Derived::integrator;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp(T* result, const T& t) const{
    assert((t*this->direction() >= this->t_old()*this->direction() && t*this->direction() <= this->interp_new_state_ptr()[0]*this->direction()) && "Out of bounds interpolation requested");
    if (this->t_old() == this->t_new()){
        ndspan::copy_array(result, this->new_state_ptr(), this->Nsys());
    }
    return interp_impl(result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::n_evals_rhs() const{
    return _n_evals_rhs;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::n_evals_jac() const{
    return _n_evals_jac;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T BaseSolver<Derived, T, N, SP, OdeType>::auto_step(T t, const T* q) const{
    //returns absolute value of emperically determined first step.
    const int dir = _direction;

    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    size_t n = this->Nsys();
    T h0, d2, h1;

    
    // ----------- Used only if N > 0 --------------
    static thread_local std::array<T, 4*N> work;
    // ---------------------------------------------

    T* y1;

    if constexpr (N > 0){
        y1 = work.data();
    } else{
        _cache_4.resize(4, this->Nsys()); // will only resize the first time.
        y1 = _cache_4.ptr(0, 0);
    }

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
BaseSolver<Derived, T, N, SP, OdeType>::Clone* BaseSolver<Derived, T, N, SP, OdeType>::clone() const {
    return new Derived(*THIS);
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
    return THIS->adv_impl();
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<OptionalObserver<T> Callable>
pbox::Box<Interpolator<T, N>> BaseSolver<Derived, T, N, SP, OdeType>::interpolate_until(const T& time, const Callable& observer){
    pbox::Box<LinkedInterpolator<T, N>> interp = pbox::make_box<LinkedInterpolator<T, N>>(this->t_old(), this->vector_old().data(), this->Nsys());
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
                interp->expand_by_owning(std::make_unique<LocalInterpolator<T, N>>(this->t(), this->vector().data(), this->Nsys()));
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
        return pbox::Box<Interpolator<T, N>>();
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
pbox::Box<Interpolator<T, N>> BaseSolver<Derived, T, N, SP, OdeType>::interp_until(const T& time, std::function<bool(const T&, const T*, const T*)> observer){
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
        throw std::runtime_error("Cannot advance until time " + to_string(time) + " because it is in the opposite direction of integration. Current time is " + to_string(this->t()) + " and direction is " + to_string(d) + ".");
    }

    constexpr bool explicit_steps = !std::is_same_v<std::decay_t<ArrayType>, EmptyArr<T>>;
    const bool has_extra_steps = explicit_steps && extra_steps.size() > 0;
    const T& t_dual = has_extra_steps ? extra_steps[extra_steps.size() - 1] : time;

    bool success;
    auto evolve = [&]() LAMBDA_INLINE -> bool {
        bool res;
        while ((res = (this->is_running() && THIS->adv_impl(time))) && (time != this->t())){
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
        throw std::runtime_error("Invalid extra steps: last extra step is " + to_string(extra_steps[extra_steps.size()-1]) + " but target time is " + to_string(time) + ". Extra steps must be in the same direction and between the current time and the target time.");
    }else{
        auto validate_idx = [&](size_t idx) LAMBDA_INLINE{
            if (extra_steps[idx]*d <= this->t()*d){
                throw std::runtime_error("Invalid extra step: " + to_string(extra_steps[idx]) + ". Extra steps must be in the same direction and between the current time (" + to_string(this->t()) + ") and the target time (" + to_string(time) + ").");
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
    ndspan::copy_array(ics+2, this->vector().data(), this->Nsys());
    return priv_apply_ics_setter(ics, t0, std::forward<Setter>(func), stepsize);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::set_ics(T t0, const T* y0, T stepsize, int direction){

    assert((direction == 1 || direction == -1 || direction == 0) && "Direction must be 1, -1, or 0");
    direction = (direction == 0) ? _direction : direction; // if 0, keep existing direction;
    if (this->validate_ics(t0, y0)){
        if (stepsize < 0) {
            std::cerr << "Cannot set negative stepsize in solver initialization" << std::endl;
            return false;
        } else if (stepsize == 0) {
            _direction = direction;
            stepsize = this->auto_step(t0, y0);
        }else{
            _direction = direction;
        }

        T* ics = const_cast<T*>(this->ics_ptr());
        ics[0] = t0;
        ics[1] = stepsize;
        ndspan::copy_array(ics+2, y0, this->Nsys());
        THIS->Reset();
        return true;
    }else {
#ifndef NO_ODE_WARN
        std::cerr << "Tried to set invalid initial conditions" << std::endl;
#endif
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::stop(const std::string& text){
    if (!this->is_running()){
        return;
    }
    _is_running = false;
    this->set_message((text == "") ? "Stopped by user" : text);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::kill(const std::string& text){
    if (this->is_dead()){
        return;
    }
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::resume(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else{
        this->set_message("Running");
        _is_running = true;
        return true;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_args(const T* new_args){
    THIS->set_args_impl(new_args);
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::unique_ptr<Interpolator<T, N>> BaseSolver<Derived, T, N, SP, OdeType>::state_interpolator(int bdr1, int bdr2) const{
    auto interp = this->local_interp();
    const T* s1 = this->old_state_ptr();
    const T* s2 = this->interp_new_state_ptr();
    return std::make_unique<CustomLocalInterpolator<T, N, decltype(interp)>>(std::move(interp), s1[0], s2[0], s1+2, s2+2, this->Nsys(), bdr1, bdr2);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
StepResult BaseSolver<Derived, T, N, SP, OdeType>::adapt_impl(T* state, const T* old_state){
    return THIS->adapt_impl(state, old_state);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp_impl(T* result, const T& t) const{
    THIS->interp_impl(result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
auto BaseSolver<Derived, T, N, SP, OdeType>::local_interp() const{
    return THIS->local_interp();
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Reset(){
    _Nupdates = 0;
    _new_state_idx = 1;
    _old_state_idx = 2;
    _true_state_idx = 1;
    _last_true_state_idx = 2;
    _aux_state_idx = 3;
    _aux2_state_idx = 4;
    _is_dead = false;
    _is_running = true;
    _diverges = false;
    _message = "Running";
    _n_evals_rhs = 0;
    _n_evals_jac = 0;
    _use_new_state = true;
    for (int i=1; i<6; i++){
        ndspan::copy_array(this->_state_data.ptr(i, 0), this->ics_ptr(), this->Nsys()+2); //copy the initial state to all others
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_args_impl(const T* new_args){
    ndspan::copy_array(_args.data(), new_args, _args.size());
    if constexpr (JP == JacPolicy::Autodiff){
        for (size_t i=0; i<_args.size(); i++){
            _args_worker[i] = DualType(new_args[i]);
        }
    }
}

//=============================================================================

// OVERRIDEN IN RICH SOLVER


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::true_state_ptr() const{
    return this->_state_data.ptr(this->_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::last_true_state_ptr() const{
    return this->_state_data.ptr(this->_last_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool BaseSolver<Derived, T, N, SP, OdeType>::adv_impl(Args&&... args){
    const int d = this->direction();
    if constexpr (sizeof...(Args) > 0){
        T time_floor = minimum_time(args...);
        if (this->is_at_new_state() && (time_floor*d <= t_new()*d)){
            return false;
        } else if (this->is_at_new_state()){
            StepResult result = this->adapt_impl(this->aux_state_ptr(), this->new_state_ptr());
            if (validate_it(result, this->aux_state_ptr())){
                register_states();
                T new_floor;
                if (THIS->RequestTimeFloor(new_floor)){
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
        StepResult result = this->adapt_impl(this->aux_state_ptr(), this->new_state_ptr());
        if (validate_it(result, this->aux_state_ptr())){
            register_states();
            T new_floor;
            if (THIS->RequestTimeFloor(new_floor) && new_floor*d < t_new()*d){
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
    return this->_state_data.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::new_state_ptr() const{
    return this->_state_data.ptr(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::old_state_ptr() const{
    return this->_state_data.ptr(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::interp_new_state_ptr() const{
    if (this->_use_new_state){
        return this->new_state_ptr();
    }else{
        return this->_state_data.ptr(5, 0); // 5th index reserved for interpolation purposes
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_new() const{
    return this->_state_data(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_old() const{
    return this->_state_data(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_last() const{
    return this->last_true_state_ptr()[0];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_message(const std::string& text){
    _message = text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::warn_paused() const{
#ifndef NO_ODE_WARN
    std::cerr << "\n" << "Solver has paused integrating. Resume before advancing." << std::endl;
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::warn_dead() const{
#ifndef NO_ODE_WARN
    std::cerr << "\n" << "Solver has permanently stopped integrating. Termination cause:\n\t" << _message << std::endl;
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::ReAdjust(const T* new_vector){
    ndspan::copy_array(this->_state_data.ptr(5, 0), this->new_state_ptr(), this->Nsys()+2); //store the re-adjusted new state for interpolation
    T* state = const_cast<T*>(this->true_state_ptr());
    state[0] = this->t();
    state[1] = this->stepsize();
    ndspan::copy_array(state+2, new_vector, this->Nsys());
    if (_true_state_idx != _new_state_idx){
        if (_last_true_state_idx == _aux_state_idx){
            _aux_state_idx = _new_state_idx;
            _new_state_idx = _true_state_idx;
        } else {
            _aux2_state_idx = _new_state_idx;
            _new_state_idx = _true_state_idx;
        }
    }
    _use_new_state = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics(T t0, const T* q0) const {
    return THIS->validate_ics_impl(t0, q0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics_impl(T t0, const T* q0) const {

    if (!all_are_finite(q0, this->Nsys()) || !isfinite(t0)){
        return false;
    }

    // ----------- Used only if N > 0 --------------
    static thread_local std::array<T, N> work;
    // ---------------------------------------------

    T* worker;
    
    if constexpr (N > 0){
        worker = work.data();
    } else{
        _cache_ics.resize(this->Nsys());
        worker = _cache_ics.data();
    }

    /*
    Calling "this", not "THIS". Derived classes that override Rhs can have their version validated.
    However since this function might be called before the Derived classes has been fully constructed,
    calling "THIS" could lead to undefined behavior.
    */
    this->Rhs(worker, t0, q0);

    return all_are_finite(worker, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_at_new_state() const{
    return _true_state_idx == _new_state_idx;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
MutView<T, Layout::F, N, N> BaseSolver<Derived, T, N, SP, OdeType>::jac_view(T* j) const{
    //returns a high level view of the jacobian matrix, so that its elements
    //can be accessed using matrix(i, j). This function simply simplifies
    //the process of constructing the correct object that can safely view the jacobian matrix
    //by doing
    // auto matrix = solver->jac_view(jac_ptr);
    // matrix(i, j) = ...
    return MutView<T, Layout::F, N, N>(j, this->Nsys(), this->Nsys());
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BaseSolver<Derived, T, N, SP, OdeType>::BaseSolver(SOLVER_CONSTRUCTOR(T)) : _state_data(6, nsys+2), _args(args.data(), args.size()), _args_worker(JP==JacPolicy::Autodiff ? args.size() : 0), _ode(ode), _Nsys(nsys), _direction(dir){
    assert(nsys > 0 && "Ode system size is 0");
    _scalar_data = {rtol, atol, min_step, max_step};
    if (stepsize < 0){
        throw std::runtime_error("The stepsize argument cannot be negative");
    }
    if (max_step < min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    if constexpr (JP == JacPolicy::Autodiff){
        for (size_t i=0; i<_args.size(); i++){
            _args_worker[i] = DualType(_args[i]);
        }
    }
    
    if (q0 == nullptr){
        this->kill("Initial conditions not set (nullptr provided)");
    } else if (this->validate_ics_impl(t0, q0)){
        T habs = (stepsize == 0 ? this->auto_step(t0, q0) : abs<T>(stepsize));
        _state_data(0, 0) = t0;
        _state_data(0, 1) = habs;
        ndspan::copy_array(_state_data.ptr(0, 2), q0, this->Nsys());
        for (int i=1; i<5; i++){
            ndspan::copy_array(this->_state_data.ptr(i, 0), this->ics_ptr(), this->Nsys()+2);
        }
    }else {
        this->kill("Initial conditions contain nan or inf, or ode(ics) does");
    }
}


// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T* BaseSolver<Derived, T, N, SP, OdeType>::aux_state_ptr() const{
    return _state_data.ptr(_aux_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T* BaseSolver<Derived, T, N, SP, OdeType>::aux_state_ptr(){
    return _state_data.ptr(_aux_state_idx, 0);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::register_states(){
    if (_old_state_idx == _last_true_state_idx){
        _old_state_idx = _new_state_idx;
        _new_state_idx = _true_state_idx = _aux_state_idx;
        _aux_state_idx = _last_true_state_idx;
        _last_true_state_idx = _old_state_idx;
    }else {
        _aux2_state_idx = _last_true_state_idx;
        _new_state_idx = _aux_state_idx;
        _aux_state_idx = _old_state_idx;
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
        _old_state_idx = _last_true_state_idx;
    }
    _use_new_state = true;
    _Nupdates++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_it(StepResult result, const T* state){
    bool success = true;
    switch (result){
        case StepResult::Success:
            break;
        case StepResult::INF_ERROR:
            this->set_message("ODE solution diverges (inf or nan encountered)");
            this->_diverges = true;
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
        _use_new_state = false;
        T* d = _state_data.ptr(5, 0);
        ndspan::copy_array(d, this->old_state_ptr(), this->Nsys()+2);
    }

    return success;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::move_state(const T& time){
    const int d = this->direction();
    assert( (time*d > this->t_last()*d && time*d <= this->t_new()*d) && "Out of bounds time requested in move_state");



    if ((time*d < this->t_new()*d)) {
        if (_true_state_idx == _new_state_idx) {
            if (_last_true_state_idx == _aux_state_idx){
                T* ptr = _state_data.ptr(_aux2_state_idx, 0);
                set_state(time, ptr);
                _true_state_idx = _aux2_state_idx;
                _aux2_state_idx = _old_state_idx;
            } else {
                T* ptr = _state_data.ptr(_aux_state_idx, 0);
                set_state(time, ptr);
                _true_state_idx = _aux_state_idx;
                _aux_state_idx = _aux2_state_idx;
                _aux2_state_idx = _old_state_idx;
            }
        }else{
            T* ptr = _state_data.ptr(_aux_state_idx, 0);
            set_state(time, ptr);
            _last_true_state_idx = _true_state_idx;
            _true_state_idx = _aux_state_idx;
            _aux_state_idx = _last_true_state_idx;
        }
    }else if (_true_state_idx != _new_state_idx){
        // update the true state to the new state, because time is exactly at t_new
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
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
        assert(all_are_finite(ics+2, this->Nsys()) && "Invalid ics in apply_ics_setter");
        if (stepsize < 0) {
            std::cerr << "Cannot set negative stepsize in solver initialization" << std::endl;
        } else if (stepsize == 0) {
            stepsize = this->auto_step(t0, ics+2);
        }
        ics[1] = stepsize;
        THIS->Reset();
    } else {
        auto res = func(ics+2);
        assert(all_are_finite(ics+2, this->Nsys()) && "Invalid ics in apply_ics_setter");
        if (stepsize < 0) {
            std::cerr << "Cannot set negative stepsize in solver initialization" << std::endl;
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