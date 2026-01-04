#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include "virtualsolver.hpp"
#include "solverstate.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T, N) OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}

#define MAIN_CONSTRUCTOR(T, N) OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args

#define SOLVER_CONSTRUCTOR(T, N) OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args

#define ODE_CONSTRUCTOR(T, N) MAIN_DEFAULT_CONSTRUCTOR(T, N), EVENTS events={}, const std::string& method="RK45"

#define ARGS ode, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args

template<typename Derived, typename T, size_t N, SolverPolicy SP>
class BaseSolver : public BaseInterface<T, N, SP>{

    using Base = BaseInterface<T, N, SP>;
    using Clone = SolverCloneType<Derived, T, N, SP>;

public:
    BaseSolver() = delete;

    // ODE PROPERTIES
    inline void                 rhs(T* dq_dt, const T& t, const T* q) const;
    inline void                 jac(T* jm, const T& t, const T* q) const;

    // ACCESSORS
    inline const T&             t() const;
    inline View1D<const T, N>   vector() const;
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
    inline State<const T>       state() const;
    inline State<const T>       ics() const;
    const std::string&          method() const;
    inline void                 interp(T* result, const T& t) const;
    T                           auto_step(T t, const T* q) const;
    T                           auto_step() const;
    Clone*                      clone() const;

    // MODIFIERS
    bool                        advance();
    bool                        advance_until(T time);
    void                        reset();
    void                        stop(const std::string& text = "");
    void                        kill(const std::string& text = "");
    bool                        resume();
    void                        set_obj(const void* obj);
    inline void                 set_args(const T* new_args);


protected:
    // =================== STATIC OVERRIDES (NECESSARY) ===============================

    static constexpr const char*    name = Derived::name;
    static constexpr bool           IS_IMPLICIT = Derived::IS_IMPLICIT;
    static constexpr int            ERR_EST_ORDER = Derived::ERR_EST_ORDER;

    inline VirtualInterp<T, N>  state_interpolator(int bdr1, int bdr2) const;
    inline void                 adapt_impl(T* state);
    inline void                 interp_impl(T* result, const T& t) const;
    //=================================================================================

    // =================== STATIC OVERRIDES (OPTIONAL) ================================
    inline void                 reset_impl();
    inline void                 set_args_impl(const T* args);
    inline void                 re_adjust_impl();
    //============= ALWAYS CALL THE BASE CLASS'S IMPL IN OPTIONAL OVERRIDE ============


    // =========================== HELPER METHODS =====================================
    inline const T*             ics_ptr() const;
    inline const T*             new_state_ptr() const;
    inline const T*             old_state_ptr() const;
    inline const T*             true_state_ptr() const;
    inline const T*             last_true_state_ptr() const;

    const T&                    t_new() const;
    const T&                    t_old() const;
    const T&                    t_last() const;
    inline void                 set_message(const std::string& text);
    void                        warn_paused() const;
    void                        warn_dead() const;
    void                        re_adjust();
    void                        remake_new_state(const T* vector);

    // ================================================================================

    //============================= OVERRIDEN IN RICH SOLVER ==========================
    inline const T&             t_impl() const;
    inline View1D<const T, N>   vector_impl() const;
    inline bool                 adv_impl();
    //=================================================================================

    DEFAULT_RULE_OF_FOUR(BaseSolver)

    BaseSolver(SOLVER_CONSTRUCTOR(T, N));
    ~BaseSolver() = default;

    T                                   MAX_FACTOR = 10;
    T                                   SAFETY = T(9)/10;
    T                                   MIN_FACTOR = T(2)/10;
    T                                   MIN_STEP = 100*std::numeric_limits<T>::epsilon();

private:

    inline const T*             aux_state_ptr() const;
    inline T*                   aux_state_ptr();
    void                        register_states();
    bool                        validate_it(const T* state);

    
    Array2D<T, 5, (N>0 ? N+2 : 0)>      _state_data;
    Array1D<T, 4, Allocation::Stack>    _scalar_data;
    Array1D<T>                          _args;
    OdeData<T>                          _ode;
    size_t                              _Nsys;
    size_t                              _Nupdates=0;
    std::string                         _message = "Running";
    std::string                         _name = name;
    int                                 _direction;
    int                                 _new_state_idx = 1;
    int                                 _old_state_idx = 2;
    int                                 _true_state_idx = 1;
    int                                 _last_true_state_idx = 2;
    int                                 _aux_state_idx = 3;
    int                                 _aux2_state_idx = 4;
    bool                                _is_dead = false;
    bool                                _diverges = false;
    bool                                _is_running = true;

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
    return _ode.rhs(dq_dt, t, q, _args.data(), _ode.obj);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::jac(T* jm, const T& t, const T* q) const{
    return _ode.jacobian(jm, t, q, _args.data(), _ode.obj);
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t() const{
    return THIS_C->t_impl();
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<const T, N> BaseSolver<Derived, T, N, SP>::vector() const{
    return THIS_C->vector_impl();
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
inline State<const T> BaseSolver<Derived, T, N, SP>::state() const{
    return State<const T>(this->true_state_ptr(), this->Nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline State<const T> BaseSolver<Derived, T, N, SP>::ics() const{
    return State<const T>(this->ics_ptr(), this->Nsys());
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
    Array1D<T> y1(n), f1(n);
    Array1D<T> scale(n);
    for (size_t i=0; i<n; i++){
        scale[i] = atol() + abs(q[i])*rtol();
    }
    Array1D<T> f0(n);
    this->rhs(f0.data(), t, q);
    T d0 = rms_norm(q, scale.data(), n);
    T d1 = rms_norm(f0.data(), scale.data(), n);
    if (d0 * 100000 < 1 || d1 * 100000 < 1){
        h0 = T(1)/1000000;
    }
    else{
        h0 = d0/d1/100;
    }
    for (size_t i=0; i<n; i++){
        y1[i] = q[i]+h0*dir*f0[i];
    }
    this->rhs(f1.data(), t+h0*dir, y1.data());
    Array1D<T> tmp(n);
    for (size_t i=0; i<n; i++){
        tmp[i] = f1[i] - f0[i];
    }
    d2 = rms_norm(tmp.data(), scale.data(), n) / h0;

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

    bool success = true;

    while (time*d > this->t_new()*d && this->is_running()){
        success = this->advance();
    }

    if (success && (time*d < this->t_new()*d)) {
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
    }else if (success){
        _last_true_state_idx = _true_state_idx;
        _true_state_idx = _new_state_idx;
    }
    return success;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
void BaseSolver<Derived, T, N, SP>::reset(){
    _Nupdates = 0;
    _new_state_idx = 1;
    _old_state_idx = 2;
    _true_state_idx = 1;
    _last_true_state_idx = 2;
    _aux_state_idx = 3;
    _aux2_state_idx = 4;
    _is_dead = false;
    _diverges = false;
    _message = "Running";
    for (int i=1; i<4; i++){
        copy_array(this->_state_data.data_ptr(i, 0), this->ics_ptr(), this->Nsys()+2); //copy the initial state to all others
    }
    THIS->reset_impl();
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
void BaseSolver<Derived, T, N, SP>::set_obj(const void* obj){
    _ode.obj = obj;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::set_args(const T* new_args){
    THIS->set_args_impl(new_args);
    copy_array(_args.data(), new_args, _args.size());
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
inline void BaseSolver<Derived, T, N, SP>::reset_impl(){}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::set_args_impl(const T* args){}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline void BaseSolver<Derived, T, N, SP>::re_adjust_impl(){}

//=============================================================================

// OVERRIDEN IN RICH SOLVER

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T& BaseSolver<Derived, T, N, SP>::t_impl() const{
    return this->_state_data(_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline View1D<const T, N> BaseSolver<Derived, T, N, SP>::vector_impl() const{
    return View1D<const T, N>{this->true_state_ptr()+2, this->Nsys()};
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
    return this->_state_data.data_ptr(this->_new_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::old_state_ptr() const{
    return this->_state_data.data_ptr(this->_old_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::true_state_ptr() const{
    return this->_state_data.data_ptr(this->_true_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::last_true_state_ptr() const{
    return this->_state_data.data_ptr(this->_last_true_state_idx, 0);
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
    std::cout << "\n" << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so." << std::endl;
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
void BaseSolver<Derived, T, N, SP>::remake_new_state(const T* vector){
    T* state = const_cast<T*>(this->new_state_ptr());
    state[0] = this->t();
    state[1] = this->stepsize();
    copy_array(state+2, vector, this->Nsys());
    this->re_adjust();
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP>
BaseSolver<Derived, T, N, SP>::BaseSolver(SOLVER_CONSTRUCTOR(T, N)) : _state_data(5, q0.size()+2), _args(args.data(), args.size()), _ode(ode), _Nsys(q0.size()), _direction(dir){

    _scalar_data = {rtol, atol, min_step, max_step};
    if (first_step < 0){
        throw std::runtime_error("The first_step argument must not be negative");
    }
    if (max_step < min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }
    
    if (!all_are_finite(q0.data(), q0.size()) || !is_finite(t0)){
        throw std::runtime_error("Non finite initial conditions");
    }

    T habs = (first_step == 0 ? this->auto_step(t0, q0.data()) : abs(first_step));
    _state_data(0, 0) = t0;
    _state_data(0, 1) = habs;
    copy_array(_state_data.data_ptr(0, 2), q0.data(), this->Nsys());
    for (int i=1; i<4; i++){
        copy_array(this->_state_data.data_ptr(i, 0), this->ics_ptr(), this->Nsys()+2);
    }
    JacMat<T, 0> tmp(this->Nsys(), this->Nsys());
    this->rhs(tmp.data(), t0, q0.data());
    if (!all_are_finite(tmp.data(), this->Nsys())){
        this->kill("Initial ode rhs is nan or inf");
    }
    if (_ode.jacobian != nullptr && (IS_IMPLICIT)){
        this->jac(tmp.data(), t0, q0.data());
        if (!all_are_finite(tmp.data(), tmp.size())){
            this->kill("Initial Jacobian is nan or inf");
        }
    }
}


// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline const T* BaseSolver<Derived, T, N, SP>::aux_state_ptr() const{
    return _state_data.data_ptr(_aux_state_idx, 0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP>
inline T* BaseSolver<Derived, T, N, SP>::aux_state_ptr(){
    return _state_data.data_ptr(_aux_state_idx, 0);
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

#endif
