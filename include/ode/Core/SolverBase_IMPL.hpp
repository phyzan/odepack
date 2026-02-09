#ifndef SOLVERBASE_IMPL_HPP
#define SOLVERBASE_IMPL_HPP

#include "SolverBase.hpp"

namespace ode{


// ODE PROPERTIES

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::Rhs(T* dq_dt, const T& t, const T* q) const{
    _ode.rhs(dq_dt, t, q, _args.data(), _ode.obj);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::rhs(T* dq_dt, const T& t, const T* q) const{
    this->Rhs(dq_dt, t, q);
    this->_n_evals_rhs++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::Jac(T* jm, const T& t, const T* q, const T* dt) const{
    if constexpr (HAS_JAC) {
        this->jac_exact(jm, t, q);
    } else if constexpr (RUNTIME_JAC_TYPE) {
        if (_ode.jacobian != nullptr){
            auto jac = reinterpret_cast<Func<T>>(_ode.jacobian);
            jac(jm, t, q, _args.data(), _ode.obj);
        }else{
            this->jac_approx(jm, t, q, dt);
        }
    }    
    else {
        this->jac_approx(jm, t, q, dt);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::jac(T* jm, const T& t, const T* q, const T* dt) const{
    this->Jac(jm, t, q, dt);
    this->_n_evals_jac++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::jac_approx(T* jm, const T& t, const T* q, const T* dt) const{
    const size_t n = this->Nsys();
    const T EPS_SQRT = sqrt(std::numeric_limits<T>::epsilon());
    const T threshold = this->atol();

    T* x1 = _dummy_state.data();
    T* x2 = _dummy_state.data() + n;
    T* f1 = _dummy_state.data() + 2*n;
    T* f2 = _dummy_state.data() + 3*n;

    copy_array(x1, q, n);
    copy_array(x2, q, n);

    for (size_t i = 0; i < n; i++) {
        // Compute step size: use provided dt or compute 
        T h_i = (dt != nullptr) ? dt[i] : EPS_SQRT * max(threshold, abs(q[i]));

        x1[i] = q[i] - h_i;
        x2[i] = q[i] + h_i;
        this->rhs(f1, t, x1);
        this->rhs(f2, t, x2);
        x1[i] = q[i];
        x2[i] = q[i];

        // Compute Jacobian column using central differences
        T* col = jm + i * n;
        T two_h = 2 * h_i;
        for (size_t j = 0; j < n; j++) {
            col[j] = (f2[j] - f1[j]) / two_h;
        }
    }
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::t() const{
    return THIS_C->t_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 View1D<T, N> BaseSolver<Derived, T, N, SP, RhsType, JacType>::vector() const{
    return View1D<T, N>(THIS_C->vector_impl(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 View1D<T, N> BaseSolver<Derived, T, N, SP, RhsType, JacType>::vector_old() const{
    return View1D<T, N>(this->old_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::stepsize() const{
    return this->_state_data(_new_state_idx, 1);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 int BaseSolver<Derived, T, N, SP, RhsType, JacType>::direction() const{
    return _direction;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::rtol() const{
    return this->_scalar_data[rtol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::atol() const{
    return this->_scalar_data[atol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::min_step() const{
    return this->_scalar_data[min_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::max_step() const{
    return this->_scalar_data[max_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const Array1D<T>& BaseSolver<Derived, T, N, SP, RhsType, JacType>::args() const{
    return _args;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 size_t BaseSolver<Derived, T, N, SP, RhsType, JacType>::Nsys() const{
    return _Nsys;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 size_t BaseSolver<Derived, T, N, SP, RhsType, JacType>::Nupdates() const{
    return _Nupdates;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::is_running() const{
    return _is_running;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::is_dead() const{
    return _is_dead;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::diverges() const{
    return _diverges;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const std::string& BaseSolver<Derived, T, N, SP, RhsType, JacType>::message() const{
    return _message;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::show_state(int prec) const{
    SolverState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message()).show(prec);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 State<T> BaseSolver<Derived, T, N, SP, RhsType, JacType>::new_state() const{
    return State<T>(this->new_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 State<T> BaseSolver<Derived, T, N, SP, RhsType, JacType>::old_state() const{
    return State<T>(this->old_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 State<T> BaseSolver<Derived, T, N, SP, RhsType, JacType>::state() const{
    return State<T>(this->true_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 State<T> BaseSolver<Derived, T, N, SP, RhsType, JacType>::ics() const{
    return State<T>(this->ics_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const std::string& BaseSolver<Derived, T, N, SP, RhsType, JacType>::method() const{
    return _name;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::interp(T* result, const T& t) const{
    assert((t*this->direction() >= this->t_old()*this->direction() && t*this->direction() <= this->interp_new_state_ptr()[0]*this->direction()) && "Out of bounds interpolation requested");
    if (this->t_old() == this->t_new()){
        copy_array(result, this->new_state_ptr(), this->Nsys());
    }
    return interp_impl(result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 size_t BaseSolver<Derived, T, N, SP, RhsType, JacType>::n_evals_rhs() const{
    return _n_evals_rhs;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
T BaseSolver<Derived, T, N, SP, RhsType, JacType>::auto_step(T t, const T* q) const{
    //returns absolute value of emperically determined first step.
    const int dir = _direction;

    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    size_t n = this->Nsys();
    T h0, d2, h1;
    T* y1 = _dummy_state.data();
    T* f1 = y1+n;
    T* scale = y1+2*n;
    T* f0 = y1+3*n;
    for (size_t i=0; i<n; i++){
        scale[i] = atol() + abs(q[i])*rtol();
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
        h1 = std::max(T(1)/1000000, h0/1000);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }
    return std::max(std::min({100*h0, h1, this->max_step()}), this->min_step());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
T BaseSolver<Derived, T, N, SP, RhsType, JacType>::auto_step() const{
    return auto_step(this->t(), this->vector().data());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
BaseSolver<Derived, T, N, SP, RhsType, JacType>::Clone* BaseSolver<Derived, T, N, SP, RhsType, JacType>::clone() const {
    return new Derived(*THIS_C);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::advance(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }
    return THIS->adv_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::advance_until(T time){
    return this->advance_until(time, VoidFunc);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
template<typename Callable>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::advance_until(T time, Callable&& observer){

    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }

    int d = this->direction();
    if (time*d <= this->t()*d) {
        return false;
    }

    bool success = this->is_running();
    bool condition = success;
    while (true){
        if ((success = this->advance()) && (condition = (time*d > this->t()*d))){
            observer(this->t(), THIS_C->vector_impl());
        }else{
            break;
        }

    }

    if (success){
        this->move_state(time);
        observer(this->t(), THIS_C->vector_impl());
        return true;
    }else{
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
template<typename ObjFun, typename Callable>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::advance_until(ObjFun&& obj_fun, T tol, int dir, Callable&& observer, T* worker){

    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }

    /*
    obj_fun : callable object with signature T obj_fun(const T& t, const T* q, const T* args, const void* obj)
    tol     : tolerance for root finding
    dir     : direction of root finding
    observer: called at every advance call
    worker: optional pointer to a preallocated array of size N to be used as temporary storage. Use this
        if obj_fun is calling some of this solver's methods that may overwrite internal dummy arrays.
        
    */
    assert((dir == 1 || dir == -1 || dir == 0) && "Invalid sign direction");

    if (worker == nullptr){
        worker = _dummy_state.data();
    }

    int factor = dir == 0 ? 1 : this->direction()*dir;

    auto ObjFunLike = [&](const T& t, const T* q) LAMBDA_INLINE {
        return obj_fun(t, q, this->args().data(), this->_ode.obj);
    };

    auto TrueObjFun = [&](const T& t) LAMBDA_INLINE{
        this->interp_impl(worker, t);
        return ObjFunLike(t, worker);
    };

    auto get_sgn = [&]() LAMBDA_INLINE {
        return factor*sgn(ObjFunLike(this->t(), this->vector().data()));
    };

    auto detected = [&](int s1, int s2) LAMBDA_INLINE{
        if (dir == 0){
            return s1*s2 <= 0;
        }else {
            return s1 < s2;
        }
    };

    int curr_dir = get_sgn();
    int old_dir = curr_dir;

    //iterate to the first step where obj_fun != 0
    //It is a very rare edge case that the code
    //will enter this loop even once
    while (curr_dir == 0 && this->advance()){
        curr_dir = old_dir = get_sgn();
        observer(this->t(), THIS_C->vector_impl());
        if (curr_dir == 0 && dir == 0){
            // If a root is found while trying to find a step
            // where the objective function is non zero (required for bisection)
            // then if the sign direction does not matter, this is success
            return true;
        }
    }

    bool success = this->is_running();
    bool condition = success;
    while (true){
        if ((success = this->advance()) && (old_dir = curr_dir, curr_dir = get_sgn(), condition = (!detected(old_dir, curr_dir)))){
            observer(this->t(), THIS_C->vector_impl());
        }else{
            break;
        }
    }

    if (success){
        T time = bisect<T, RootPolicy::Right>(TrueObjFun, this->t_last(), this->t(), tol);
        this->move_state(time);
        observer(this->t(), THIS_C->vector_impl());
        return true;
    }else{
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::observe_until(T time, std::function<void(const T&, const T*)> observer){
    return this->advance_until(time, observer);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::reset(){
    THIS->reset_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
template<typename Setter>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::apply_ics_setter(T t0, Setter&& func, T stepsize){
    T* ics = const_cast<T*>(this->ics_ptr());
    ics[0] = t0;
    func(ics+2);
    assert(all_are_finite(ics+2, this->Nsys()) && "Invalid ics in apply_ics_setter");
    if (stepsize < 0) {
        std::cerr << "Cannot set negative stepsize in solver initialization" << std::endl;
    } else if (stepsize == 0) {
        stepsize = this->auto_step(t0, ics+2);
    }
    ics[1] = stepsize;
    this->reset();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_ics(T t0, const T* y0, T stepsize, int direction){

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
        copy_array(ics+2, y0, this->Nsys());
        this->reset();
        return true;
    }else {
        std::cerr << "Tried to set invalid initial conditions" << std::endl;
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::stop(const std::string& text){
    if (!this->is_running()){
        return;
    }
    _is_running = false;
    this->set_message((text == "") ? "Stopped by user" : text);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::kill(const std::string& text){
    if (this->is_dead()){
        return;
    }
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::resume(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else{
        this->set_message("Running");
        _is_running = true;
        return true;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_obj(const void* obj){
    assert(obj != this && "Cannot set obj equal to the pointer of the solver, as that may cause UB when the solver is copied/moved");
    _ode.obj = obj;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_args(const T* new_args){
    THIS->set_args_impl(new_args);
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 std::unique_ptr<Interpolator<T, N>> BaseSolver<Derived, T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    return THIS_C->state_interpolator(bdr1, bdr2);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::adapt_impl(T* state){
    THIS->adapt_impl(state);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    THIS_C->interp_impl(result, t);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::jac_exact(T* jm, const T& t, const T* q) const{
    if constexpr (RUNTIME_JAC_TYPE) {
        auto jac = reinterpret_cast<Func<T>>(_ode.jacobian);
        assert(jac != nullptr && "Jacobian function pointer is null");
        jac(jm, t, q, _args.data(), _ode.obj);
    } else if constexpr (HAS_JAC) {
        if constexpr (std::is_pointer_v<JacType>) {
            assert(_ode.jacobian != nullptr && "Jacobian function pointer is null");
        }
        _ode.jacobian(jm, t, q, _args.data(), _ode.obj);
    } else {
        throw std::runtime_error("Jacobian function not provided for this solver.");
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::reset_impl(){
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
        copy_array(this->_state_data.ptr(i, 0), this->ics_ptr(), this->Nsys()+2); //copy the initial state to all others
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_args_impl(const T* new_args){
    copy_array(_args.data(), new_args, _args.size());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::re_adjust_impl(const T* new_vector){}

//=============================================================================

// OVERRIDEN IN RICH SOLVER

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::t_impl() const{
    return this->_state_data(_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::vector_impl() const{
    return this->true_state_ptr()+2;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::adv_impl(){
    if (_true_state_idx == _new_state_idx){
        this->adapt_impl(this->aux_state_ptr());
        if (validate_it(this->aux_state_ptr())){
            register_states();
            this->_Nupdates++;
            return true;
        }else{
            return false;
        }
    }
    else{
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
        return true;
    }
}

// HELPER METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::ics_ptr() const{
    return this->_state_data.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::new_state_ptr() const{
    return this->_state_data.ptr(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::old_state_ptr() const{
    return this->_state_data.ptr(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::true_state_ptr() const{
    return this->_state_data.ptr(this->_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::last_true_state_ptr() const{
    return this->_state_data.ptr(this->_last_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::interp_new_state_ptr() const{
    if (this->_use_new_state){
        return this->new_state_ptr();
    }else{
        return this->_state_data.ptr(5, 0); // 5th index reserved for interpolation purposes
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::t_new() const{
    return this->_state_data(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::t_old() const{
    return this->_state_data(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T& BaseSolver<Derived, T, N, SP, RhsType, JacType>::t_last() const{
    return this->_state_data(this->_last_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_message(const std::string& text){
    _message = text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::warn_paused() const{
#ifndef NO_ODE_WARN
    std::cerr << "\n" << "Solver has paused integrating. Resume before advancing." << std::endl;
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::warn_dead() const{
#ifndef NO_ODE_WARN
    std::cerr << "\n" << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << std::endl;
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::re_adjust(const T* new_vector){
    THIS->re_adjust_impl(new_vector);
    copy_array(this->_state_data.ptr(5, 0), this->new_state_ptr(), this->Nsys()+2); //store the re-adjusted new state for interpolation
    T* state = const_cast<T*>(this->new_state_ptr());
    state[0] = this->t();
    state[1] = this->stepsize();
    copy_array(state+2, new_vector, this->Nsys());
    _true_state_idx = _new_state_idx;
    _use_new_state = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::validate_ics(T t0, const T* q0) const {
    return THIS_C->validate_ics_impl(t0, q0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::validate_ics_impl(T t0, const T* q0) const {

    if (!all_are_finite(q0, this->Nsys()) || !is_finite(t0)){
        return false;
    }

    this->rhs(_dummy_state.data(), t0, q0);
    return all_are_finite(_dummy_state.data(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::is_at_new_state() const{
    return _true_state_idx == _new_state_idx;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
MutView<T, Layout::F, N, N> BaseSolver<Derived, T, N, SP, RhsType, JacType>::jac_view(T* j) const{
    //returns a high level view of the jacobian matrix, so that its elements
    //can be accessed using matrix(i, j). This function simply simplifies
    //the process of constructing the correct object that can safely view the jacobian matrix
    //by doing
    // auto matrix = solver->jac_view(jac_ptr);
    // matrix(i, j) = ...
    return MutView<T, Layout::F, N, N>(j, this->Nsys(), this->Nsys());
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
BaseSolver<Derived, T, N, SP, RhsType, JacType>::BaseSolver(SOLVER_CONSTRUCTOR(T)) : _state_data(6, nsys+2), _dummy_state(4*nsys), _args(args.data(), args.size()), _ode(ode), _Nsys(nsys), _direction(dir){
    if constexpr (std::is_pointer_v<RhsType>){
        assert(ode.rhs != nullptr && "RHS function pointer cannot be nullptr");
    }
    if constexpr (std::is_pointer_v<JacType>){
        assert(ode.jacobian != nullptr && "Explicitly passed Jacobian function pointer cannot be nullptr");
    }
    assert(nsys > 0 && "Ode system size is 0");
    _scalar_data = {rtol, atol, min_step, max_step};
    if (stepsize < 0){
        throw std::runtime_error("The stepsize argument cannot be negative");
    }
    if (max_step < min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }
    if (q0 == nullptr){
        this->kill("Initial conditions not set (nullptr provided)");
    }else if (this->validate_ics_impl(t0, q0)){
        T habs = (stepsize == 0 ? this->auto_step(t0, q0) : abs(stepsize));

        _state_data(0, 0) = t0;
        _state_data(0, 1) = habs;
        copy_array(_state_data.ptr(0, 2), q0, this->Nsys());
        for (int i=1; i<5; i++){
            copy_array(this->_state_data.ptr(i, 0), this->ics_ptr(), this->Nsys()+2);
        }
    }else {
        this->kill("Initial conditions contain nan or inf, or ode(ics) does");
    }
}


// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 const T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::aux_state_ptr() const{
    return _state_data.ptr(_aux_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 T* BaseSolver<Derived, T, N, SP, RhsType, JacType>::aux_state_ptr(){
    return _state_data.ptr(_aux_state_idx, 0);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::register_states(){
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
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
bool BaseSolver<Derived, T, N, SP, RhsType, JacType>::validate_it(const T* state){
    bool success = true;
    if (this->is_dead()){
        // The derived adapt_impl may kill the solver under the conditions that it deems so.
        success = false;
    }
    else if (!all_are_finite(state+2, this->Nsys())){
        this->kill("Ode solution diverges");
        this->_diverges = true;
        success = false;
    }
    else if (state[1] <= MIN_STEP){
        this->kill("Required stepsize was smaller than machine precision");
        success = false;
    }
    else if (state[0] == this->t_new()){
        this->kill("The next time step is identical to the previous one, possibly due to machine rounding error");
        success = false;
    }

    if (!success){
        //close the interpolation interval as most integration algorithms
        //alter their interpolation polynomials when calling adapt_impl,
        //but since the step failed, the current interpolation interval is no longer valid.
        _use_new_state = false;
        T* d = _state_data.ptr(5, 0);
        copy_array(d, this->old_state_ptr(), this->Nsys()+2);
    }

    return success;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::update_state(const T& time){
    assert( (time*this->direction() > this->t_last()*this->direction() && time*this->direction() <= this->t_new()*this->direction()) && "Out of bounds time requested in update_state");
    if ((time*this->direction() < this->t_new()*this->direction())) {
        T* ptr = this->aux_state_ptr();
        set_state(time, ptr);
        if (this->t_impl() != this->t_new()) {
            _last_true_state_idx = _true_state_idx;
            _true_state_idx = _aux_state_idx;
            _aux_state_idx = _aux2_state_idx = _last_true_state_idx;
        }else {
            _true_state_idx = _aux_state_idx;
            _aux_state_idx = _aux2_state_idx;
        }
    }else if (_true_state_idx != _new_state_idx){
        // update the true state to the new state, because time is exactly at t_new
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
    }

}


template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void BaseSolver<Derived, T, N, SP, RhsType, JacType>::move_state(const T& time){
    assert( (time*this->direction() > this->t_last()*this->direction() && time*this->direction() <= this->t_new()*this->direction()) && "Out of bounds time requested in move_state");


    if ((time*this->direction() < this->t_new()*this->direction())) {
        if (this->t_impl() == this->t_new()) {
            int idx = _last_true_state_idx == _aux_state_idx ? _aux2_state_idx : _aux_state_idx;
            T* ptr = _state_data.ptr(idx, 0);
            set_state(time, ptr);
            _true_state_idx = idx;
        }else{
            T* ptr = const_cast<T*>(this->true_state_ptr());
            set_state(time, ptr);
        }
    }else if (_true_state_idx != _new_state_idx){
        // update the true state to the new state, because time is exactly at t_new
        _aux_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
 void BaseSolver<Derived, T, N, SP, RhsType, JacType>::set_state(const T& time, T* state){
    state[0] = time;
    state[1] = this->stepsize();
    interp(state+2, time);
}

} // namespace ode

#endif // SOLVERBASE_IMPL_HPP