#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include "tools.hpp"
#include "virtualsolver.hpp"
#include "solverstate.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T) OdeData<T> ode, const T& t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}

#define MAIN_CONSTRUCTOR(T) OdeData<T> ode, const T& t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args

#define SOLVER_CONSTRUCTOR(T) OdeData<T> ode, const T& t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args

#define ODE_CONSTRUCTOR(T) MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events={}, const std::string& method="RK45"

#define ARGS ode, t0, q0, nsys, rtol, atol, min_step, max_step, first_step, dir, args

template<typename Derived, typename T, size_t N, SolverPolicy SP>
class BaseSolver : public BaseInterface<T, N, SP>{

    using Base = BaseInterface<T, N, SP>;
    using Clone = SolverCloneType<Derived, T, N, SP>;

public:

    using Scalar = T;
    static constexpr size_t NSYS = N;
    static constexpr SolverPolicy Policy = SP;
    
    BaseSolver() = delete;

    // ODE PROPERTIES
    inline void                 rhs(T* dq_dt, const T& t, const T* q) const;
    inline void                 jac(T* jm, const T& t, const T* q, const T* dt = nullptr) const;
    inline void                 jac_approx(T* j, const T& t, const T* q, const T* dt) const;
    MutView<T, Layout::F, N, N> jac_view(T* j) const;

    // ACCESSORS
    inline const T&             t() const;
    const T&                    t_old() const;
    inline View1D<T, N>         vector() const;
    inline View1D<T, N>         vector_old() const;
    inline const T&             stepsize() const;
    inline int                  direction() const;
    inline const T&             rtol() const;
    inline const T&             atol() const;
    inline const T&             min_step() const;
    inline const T&             max_step() const;
    inline const Array1D<T>&    args() const;
    inline size_t               Nsys() const;
    inline size_t               Nupdates() const;
    inline bool                 is_running() const;
    inline bool                 is_dead() const;
    inline bool                 diverges() const;
    inline const std::string&   message() const;
    void                        show_state(int prec=8) const;
    inline State<T>             new_state() const;
    inline State<T>             old_state() const;
    inline State<T>             state() const;
    inline State<T>             ics() const;
    bool                        validate_ics(T t0, const T* q0) const;
    const std::string&          method() const;
    inline void                 interp(T* result, const T& t) const;
    T                           auto_step(T t, const T* q) const;
    T                           auto_step() const;
    Clone*                      clone() const;

    // MODIFIERS
    bool                        advance();
    bool                        advance_until(T time);

    template<typename Callable>
    bool                        advance_until(Callable&& obj_fun, T tol, int dir=0, T* worker = nullptr);
    void                        reset();

    template<typename Setter>
    void                        apply_ics_setter(T t0, Setter&& func, T stepsize = 0);

    bool                        set_ics(T t0, const T* y0, T stepsize = 0); //resets the solver with new ics without reallocating
    void                        stop(const std::string& text = "");
    void                        kill(const std::string& text = "");
    bool                        resume();
    inline void                 set_obj(const void* obj);
    inline void                 set_args(const T* new_args);


protected:
    // =================== STATIC OVERRIDES (NECESSARY) ===============================

    static constexpr const char*    name = Derived::name;
    static constexpr bool           IS_IMPLICIT = Derived::IS_IMPLICIT;
    static constexpr int            ERR_EST_ORDER = Derived::ERR_EST_ORDER;

    inline VirtualInterp<T, N>  state_interpolator(int bdr1, int bdr2) const;
    inline void                 adapt_impl(T* state);
    inline void                 interp_impl(T* result, const T& t) const;
    // ================================================================================

    // ========================= STATIC OVERRIDES (OPTIONAL) ==========================
    inline void                 rhs_impl(T* dq_dt, const T& t, const T* q) const;
    inline void                 jac_impl(T* jm, const T& t, const T* q, const T* dt) const;
    inline void                 reset_impl();                                   //call base class's override first
    inline void                 set_args_impl(const T* new_args);               //call base class's override first
    inline void                 re_adjust_impl();                               //call base class's override first
    inline bool                 validate_ics_impl(T t0, const T* q0) const;     //call base class's override first
    // ================================================================================


    // =========================== HELPER METHODS =====================================
    inline const T*             ics_ptr() const;
    inline const T*             new_state_ptr() const;
    inline const T*             old_state_ptr() const;
    inline const T*             true_state_ptr() const;
    inline const T*             last_true_state_ptr() const;

    const T&                    t_new() const;
    public:
    const T&                    t_last() const;
    void                        warn_paused() const;
    void                        warn_dead() const;
    inline void                 set_message(const std::string& text);
    void                        re_adjust();
    void                        remake_new_state(const T* vector);


    // ================================================================================

    // ============================ OVERRIDEN IN RICH SOLVER ==========================
    inline const T&             t_impl() const;
    inline View1D<T, N>         vector_impl() const;
    inline bool                 adv_impl();
    // ================================================================================

    DEFAULT_RULE_OF_FOUR(BaseSolver)

    BaseSolver(SOLVER_CONSTRUCTOR(T));
    ~BaseSolver() = default;

    T                                   MAX_FACTOR = 10;
    T                                   SAFETY = T(9)/10;
    T                                   MIN_FACTOR = T(2)/10;
    T                                   MIN_STEP = 100*std::numeric_limits<T>::epsilon();

private:

    inline void                 jac_exact(T* j, const T& t, const T* q) const;
    inline const T*             aux_state_ptr() const;
    inline T*                   aux_state_ptr();
    void                        register_states();
    bool                        validate_it(const T* state);
    bool                        settle_on(const T& time, bool success);


    Array2D<T, 5, (N>0 ? N+2 : 0), Allocation::Auto>    _state_data;
    mutable Array1D<T, 4*N, Allocation::Auto>           _dummy_state;
    Array1D<T, 4, Allocation::Stack>                    _scalar_data;
    Array1D<T>                                          _args;
    OdeData<T>                                          _ode;
    size_t                                              _Nsys = N;
    size_t                                              _Nupdates = 0;
    std::string                                         _message = "Running";
    std::string                                         _name = name;
    int                                                 _direction = 1;
    int                                                 _new_state_idx = 1;
    int                                                 _old_state_idx = 2;
    int                                                 _true_state_idx = 1;
    int                                                 _last_true_state_idx = 2;
    int                                                 _aux_state_idx = 3;
    int                                                 _aux2_state_idx = 4;
    bool                                                _is_dead = false;
    bool                                                _diverges = false;
    bool                                                _is_running = true;

    static constexpr int rtol_idx = 0;
    static constexpr int atol_idx = 1;
    static constexpr int min_step_idx = 2;
    static constexpr int max_step_idx = 3;
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

// ODE PROPERTIES

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::rhs(T* dq_dt, const T& t, const T* q) const{
    THIS_C->rhs_impl(dq_dt, t, q);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::jac(T* jm, const T& t, const T* q, const T* dt) const{
    THIS_C->jac_impl(jm, t, q, dt);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::jac_approx(T* jm, const T& t, const T* q, const T* dt) const{
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
        // Compute step size: use provided dt or compute inline
        T h_i = (dt != nullptr) ? dt[i] : EPS_SQRT * std::max(threshold, abs(q[i]));

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

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t() const{
    return THIS_C->t_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<T, N> BaseSolver<Derived, T, N, SP>::vector() const{
    return THIS_C->vector_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<T, N> BaseSolver<Derived, T, N, SP>::vector_old() const{
    return View1D<T, N>(this->old_state_ptr()+2, this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::stepsize() const{
    return this->_state_data(_new_state_idx, 1);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline int BaseSolver<Derived, T, N, SP>::direction() const{
    return _direction;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::rtol() const{
    return this->_scalar_data[rtol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::atol() const{
    return this->_scalar_data[atol_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::min_step() const{
    return this->_scalar_data[min_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::max_step() const{
    return this->_scalar_data[max_step_idx];
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const Array1D<T>& BaseSolver<Derived, T, N, SP>::args() const{
    return _args;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline size_t BaseSolver<Derived, T, N, SP>::Nsys() const{
    return _Nsys;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline size_t BaseSolver<Derived, T, N, SP>::Nupdates() const{
    return _Nupdates;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool BaseSolver<Derived, T, N, SP>::is_running() const{
    return _is_running;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool BaseSolver<Derived, T, N, SP>::is_dead() const{
    return _is_dead;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool BaseSolver<Derived, T, N, SP>::diverges() const{
    return _diverges;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const std::string& BaseSolver<Derived, T, N, SP>::message() const{
    return _message;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::show_state(int prec) const{
    SolverState<T, N>(this->vector().data(), this->t(), this->stepsize(), this->Nsys(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message()).show(prec);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline State<T> BaseSolver<Derived, T, N, SP>::new_state() const{
    return State<T>(this->new_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline State<T> BaseSolver<Derived, T, N, SP>::old_state() const{
    return State<T>(this->old_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline State<T> BaseSolver<Derived, T, N, SP>::state() const{
    return State<T>(this->true_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline State<T> BaseSolver<Derived, T, N, SP>::ics() const{
    return State<T>(this->ics_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const std::string& BaseSolver<Derived, T, N, SP>::method() const{
    return _name;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::interp(T* result, const T& t) const{
    int d = this->direction();
    if (t*d < this->t_old()*d || this->t_new()*d < t*d ){
        throw std::runtime_error("Cannot perform local interpolation at t = " + to_string(t) + " between the states t_1 = " + to_string(this->t_old()) + " and t_2 = " + to_string(this->t_new()));
    }
    return interp_impl(result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
T BaseSolver<Derived, T, N, SP>::auto_step(T t, const T* q) const{
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

template<typename Derived, typename T, size_t N, SolverPolicy SP>
T BaseSolver<Derived, T, N, SP>::auto_step() const{
    return auto_step(this->t(), this->vector().data());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
BaseSolver<Derived, T, N, SP>::Clone* BaseSolver<Derived, T, N, SP>::clone() const {
    return new Derived(*THIS_C);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::advance(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else if (!this->is_running()) {
        this->warn_paused();
        return false;
    }
    return THIS->adv_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::advance_until(T time){
    int d = this->direction();
    if (time*d <= this->t()*d) {
        return false;
    }

    bool success = this->is_running();
    while (time*d > this->t_new()*d && this->is_running()){
        success = this->advance();
    }

    return this->settle_on(time, success);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
template<typename Callable>
bool BaseSolver<Derived, T, N, SP>::advance_until(Callable&& obj_fun, T tol, int dir, T* worker){

    /*
    obj_fun : callable object with signature T obj_fun(const T& t, const T* q, const T* args, const void* obj)
    tol     : tolerance for root finding
    dir     : direction of root finding
    worker: optional pointer to a preallocated array of size N to be used as temporary storage. Use this
        if obj_fun is calling some of this solver's methods that may overwrite internal dummy arrays.
    */
    assert((dir == 1 || dir == -1 || dir == 0) && "Invalid sign direction");

    if (worker == nullptr){
        worker = _dummy_state.data();
    }

    int factor = dir == 0 ? 1 : this->direction()*dir;

    auto ObjFun = [&](const T& t, const T* q) LAMBDA_INLINE {
        return obj_fun(t, q, this->args().data(), this->_ode.obj);
    };

    auto TrueObjFun = [&](const T& t) LAMBDA_INLINE{
        this->interp_impl(worker, t);
        return ObjFun(t, worker);
    };

    auto get_sgn = [&]() LAMBDA_INLINE {
        return factor*sgn(ObjFun(this->t(), this->vector().data()));
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
    }

    bool success = this->is_running();
    while ( !detected(old_dir, curr_dir) && this->is_running()){
        success = this->advance();
        old_dir = curr_dir;
        curr_dir = get_sgn();
    }

    T time;
    if (success){
        time = bisect<T, RootPolicy::Right>(TrueObjFun, this->t_last(), this->t(), tol);
    }

    return settle_on(time, success);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::reset(){
    THIS->reset_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
template<typename Setter>
void BaseSolver<Derived, T, N, SP>::apply_ics_setter(T t0, Setter&& func, T stepsize){
    T* ics = const_cast<T*>(this->ics_ptr());
    ics[0] = t0;
    func(ics+2);
    assert(all_are_finite(ics+2, this->Nsys()) && "Invalid ics in apply_ics_setter");
    if (stepsize < 0) {
        throw std::runtime_error("Cannot set negative stepsize in solver initialization");
    } else if (stepsize == 0) {
        stepsize = this->auto_step(t0, ics+2);
    }
    ics[1] = stepsize;
    this->reset();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::set_ics(T t0, const T* y0, T stepsize){

    if (this->validate_ics(t0, y0)){
        if (stepsize < 0) {
            throw std::runtime_error("Cannot set negative stepsize in solver initialization");
        } else if (stepsize == 0) {
            stepsize = this->auto_step(t0, y0);
        }

        T* ics = const_cast<T*>(this->ics_ptr());
        ics[0] = t0;
        ics[1] = stepsize;
        copy_array(ics+2, y0, this->Nsys());
        this->reset();
        return true;
    }else {
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::stop(const std::string& text){
    if (!this->is_running()){
        return;
    }
    _is_running = false;
    this->set_message((text == "") ? "Stopped by user" : text);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::kill(const std::string& text){
    if (this->is_dead()){
        return;
    }
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::resume(){
    if (this->is_dead()){
        this->warn_dead();
        return false;
    }else{
        this->set_message("Running");
        _is_running = true;
        return true;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::set_obj(const void* obj){
    assert(obj != this && "Cannot set obj equal to the pointer of the solver, as that may cause UB when the solver is copied/moved");
    _ode.obj = obj;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::set_args(const T* new_args){
    THIS->set_args_impl(new_args);
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline std::unique_ptr<Interpolator<T, N>> BaseSolver<Derived, T, N, SP>::state_interpolator(int bdr1, int bdr2) const{
    return THIS_C->state_interpolator(bdr1, bdr2);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::adapt_impl(T* state){
    THIS->adapt_impl(state);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::interp_impl(T* result, const T& t) const{
    THIS_C->interp_impl(result, t);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::rhs_impl(T* dq_dt, const T& t, const T* q) const{
    assert(_ode.rhs != nullptr && "The ode rhs provided is a null pointer, or has not been properly overriden");
    _ode.rhs(dq_dt, t, q, _args.data(), _ode.obj);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::jac_impl(T* jm, const T& t, const T* q, const T* dt) const{
    if (_ode.jacobian != nullptr){
        this->jac_exact(jm, t, q);
    }else{
        this->jac_approx(jm, t, q, dt);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::jac_exact(T* jm, const T& t, const T* q) const{
    assert(_ode.jacobian != nullptr && "Jacobian is null");
    _ode.jacobian(jm, t, q, _args.data(), _ode.obj);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::reset_impl(){
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
    for (int i=1; i<5; i++){
        copy_array(this->_state_data.ptr(i, 0), this->ics_ptr(), this->Nsys()+2); //copy the initial state to all others
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::set_args_impl(const T* new_args){
    copy_array(_args.data(), new_args, _args.size());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::re_adjust_impl(){}

//=============================================================================

// OVERRIDEN IN RICH SOLVER

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t_impl() const{
    return this->_state_data(_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<T, N> BaseSolver<Derived, T, N, SP>::vector_impl() const{
    return View1D<T, N>{this->true_state_ptr()+2, this->Nsys()};
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline bool BaseSolver<Derived, T, N, SP>::adv_impl(){
    if (_true_state_idx == _new_state_idx){
        this->adapt_impl(this->aux_state_ptr());
        if (validate_it(this->aux_state_ptr())){
            register_states();
            this->_Nupdates++;
            return true;
        }
        else{
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

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::ics_ptr() const{
    return this->_state_data.data();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::new_state_ptr() const{
    return this->_state_data.ptr(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::old_state_ptr() const{
    return this->_state_data.ptr(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::true_state_ptr() const{
    return this->_state_data.ptr(this->_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::last_true_state_ptr() const{
    return this->_state_data.ptr(this->_last_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t_new() const{
    return this->_state_data(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t_old() const{
    return this->_state_data(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t_last() const{
    return this->_state_data(this->_last_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::set_message(const std::string& text){
    _message = text;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::warn_paused() const{
    std::cout << "\n" << "Solver has paused integrating. Resume before advancing." << std::endl;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::warn_dead() const{
    std::cout << "\n" << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << std::endl;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::re_adjust(){
    THIS->re_adjust_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::validate_ics(T t0, const T* q0) const {
    return THIS_C->validate_ics_impl(t0, q0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::validate_ics_impl(T t0, const T* q0) const {

    if (!all_are_finite(q0, this->Nsys()) || !is_finite(t0)){
        return false;
    }

    this->rhs(_dummy_state.data(), t0, q0);
    return all_are_finite(_dummy_state.data(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::remake_new_state(const T* vector){
    T* state = const_cast<T*>(this->new_state_ptr());
    state[0] = this->t();
    state[1] = this->stepsize();
    copy_array(state+2, vector, this->Nsys());
    this->re_adjust();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
MutView<T, Layout::F, N, N> BaseSolver<Derived, T, N, SP>::jac_view(T* j) const{
    //returns a high level view of the jacobian matrix, so that its elements
    //can be accessed using matrix(i, j). This function simply simplifies
    //the process of constructing the correct object that can safely view the jacobian matrix
    //by doing
    // auto matrix = solver->jac_view(jac_ptr);
    // matrix(i, j) = ...
    return MutView<T, Layout::F, N, N>(j, this->Nsys(), this->Nsys());
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP>
BaseSolver<Derived, T, N, SP>::BaseSolver(SOLVER_CONSTRUCTOR(T)) : _state_data(5, nsys+2), _dummy_state(4*nsys), _args(args.data(), args.size()), _ode(ode), _Nsys(nsys), _direction(dir){
    assert(nsys > 0 && "Ode system size is 0");
    _scalar_data = {rtol, atol, min_step, max_step};
    if (first_step < 0){
        throw std::runtime_error("The first_step argument must not be negative");
    }
    if (max_step < min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }
    if (q0 == nullptr){
        this->kill("Initial conditions not set (nullptr provided)");
    }else if (this->validate_ics_impl(t0, q0)){
        T habs = (first_step == 0 ? this->auto_step(t0, q0) : abs(first_step));

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

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::aux_state_ptr() const{
    return _state_data.ptr(_aux_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline T* BaseSolver<Derived, T, N, SP>::aux_state_ptr(){
    return _state_data.ptr(_aux_state_idx, 0);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::register_states(){
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
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::validate_it(const T* state){
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

    return success;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
bool BaseSolver<Derived, T, N, SP>::settle_on(const T& time, bool success){
    if (success && (time*this->direction() < this->t_new()*this->direction())) {
        T* ptr = this->aux_state_ptr();
        interp(ptr+2, time);
        ptr[0] = time;
        ptr[1] = this->stepsize();
        if (this->t() != this->t_new()) {
            _last_true_state_idx = _true_state_idx;
            _true_state_idx = _aux_state_idx;
            _aux_state_idx = _aux2_state_idx = _last_true_state_idx;
        }else {
            _true_state_idx = _aux_state_idx;
            _aux_state_idx = _aux2_state_idx;
        }
    }else if (success && _true_state_idx != _new_state_idx){
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
    }
    return success;
}

#endif
