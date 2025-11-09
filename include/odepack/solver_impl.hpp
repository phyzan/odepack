#ifndef SOLVER_IMPL_HPP
#define SOLVER_IMPL_HPP

#include "odesolvers.hpp"

#define MAIN_DEFAULT_CONSTRUCTOR(T, N) OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args={}, const std::vector<const Event<T, N>*>& events={}

#define MAIN_CONSTRUCTOR(T, N) OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args, const std::vector<const Event<T, N>*>& events

#define SOLVER_CONSTRUCTOR(T, N) std::string name, OdeData<T> ode, const T& t0, const Array1D<T, N>& q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args, const std::vector<const Event<T, N>*>& events

#define ODE_CONSTRUCTOR(T, N) MAIN_DEFAULT_CONSTRUCTOR(T, N), const std::string& method="RK45"

#define ARGS ode, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events

template<typename T, size_t N, typename Derived>
inline void interp_func(T* res, const T& t, const void* obj);

template<typename T, size_t N, typename Derived>
class DerivedSolver : public OdeSolver<T, N>{

    using Solver = DerivedSolver<T, N, Derived>;
    using UniqueClone = std::unique_ptr<OdeSolver<T, N>>;

public:

    static const T MAX_FACTOR;
    static const T SAFETY;
    static const T MIN_FACTOR;
    static const T MIN_STEP;
    static constexpr bool IS_IMPLICIT = Derived::IS_IMPLICIT;

    DerivedSolver() = delete;

    DEFAULT_RULE_OF_FOUR(DerivedSolver);

    ~DerivedSolver() = default;

    const T&                                t() const final;
    const Array1D<T, N>&                    q() const final;
    const Array1D<T, N>&                    true_vector() const final;
    const T&                                stepsize() const final;
    int                                     direction() const final;
    const T&                                rtol() const final;
    const T&                                atol() const final;
    const T&                                min_step() const final;
    const T&                                max_step() const final;
    const std::vector<T>&                   args() const final;
    size_t                                  Nsys() const final;
    size_t                                  Nupdates() const final;
    bool                                    diverges() const final;
    bool                                    is_running() const final;
    bool                                    is_dead() const final;
    std::string                             message() const final;
    SolverState<T, N>                       state() const final;
    std::vector<const Event<T, N>*>         current_events() const final;
    inline const EventCollection<T, N>&     event_col() const final;
    std::string                             name() const final;
    T                                       auto_step(const ICS<T, N>* ics = nullptr) const final;
    OdeSolver<T, N>*                        clone() const final;
    inline const Interpolator<T, N>*        interpolator() const final;
    inline bool                             is_interpolating() const final;
    inline bool                             at_event() const final;
    inline const State<T, N>&               ics() const final;

    bool                                    advance() final;
    bool                                    advance_to_event() final;
    void                                    stop(std::string text) final;
    void                                    kill(std::string text) final;
    bool                                    resume() final;
    void                                    set_tmax(T tmax) final;
    void                                    start_interpolation() final;
    void                                    stop_interpolation() final;
    void                                    reset() override;//virtual override
    void                                    set_obj(const void* obj) final;

    inline const State<T, N>&               old_state() const;

    inline const State<T, N>&               current_state() const;

    inline void                             adapt_impl(State<T, N>& result);//virtual

    inline std::unique_ptr<Interpolator<T, N>> state_interpolator(int bdr1, int bdr2) const;//virtual. return a dynamically allocated object

    inline void                             interp(T* result, const T& t) const;//virtual

    inline void                             re_adjust();//virtual


protected:

    DerivedSolver(SOLVER_CONSTRUCTOR(T, N));

    inline void                    _rhs(T* result, const T& t, const T* q) const;

    inline void                    _jac(JacMat<T, N>& result, const T& t, const Array1D<T, N>& q) const;

private:

    void                    _register_states();

    void                    _initialize_events(const T& t0);

    void                    _add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant);

    bool                    _validate_it(const State<T, N>& state);

    void                    _warn_dead() const;

    void                    _warn_paused() const;

    inline bool             _requires_new_start() const;

    inline bool             _equiv_states() const;

    OdeData<T>                                  _ode;
    T                                           _rtol;
    T                                           _atol;
    T                                           _min_step;
    T                                           _max_step;
    std::vector<T>                              _args;
    size_t                                      _n; //size of ode system
    std::string                                 _name;
    bool                                        _diverges = false;
    bool                                        _is_running = true;
    bool                                        _is_dead = false;
    size_t                                      _N=0;//total number of solution updates
    std::string                                 _message = "Running";
    EventCollection<T, N>                       _events;
    long int                                    _event_idx = -1;
    int                                         _direction;
    std::array<State<T, N>, 4>                  _states;
    size_t                                      _old_state_idx = 1;
    size_t                                      _curr_state_idx = 2;
    size_t                                      _aux_state_idx = 3;
    bool                                        _interp_data = false;
    LinkedInterpolator<T, N>                    _current_linked_interpolator;

};


template<typename T, size_t N, typename Derived>
DerivedSolver<T, N, Derived>::DerivedSolver(SOLVER_CONSTRUCTOR(T, N)): OdeSolver<T, N>(), _ode(ode), _rtol(rtol), _atol(atol), _min_step(std::max(min_step, MIN_STEP)), _max_step(max_step), _args(args), _n(q0.size()), _name(std::move(name)), _events(events), _direction(dir == 0 ? 1 : sgn(dir)), _current_linked_interpolator(t0, q0){
    if (first_step < 0){
        throw std::runtime_error("The first_step argument must not be negative");
    }
    if (_max_step < _min_step){
        throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
    }

    if (!all_are_finite(q0.data(), q0.size()) || !is_finite(t0)){
        throw std::runtime_error("Non finite initial conditions");
    }
    ICS<T, N> ics = {t0, q0};
    T habs = (first_step == 0 ? this->auto_step(&ics) : abs(first_step));
    for (State<T, N>& state : _states){
        state.habs = habs;
        state.t = t0;
        state.vector = q0;
    }
    
    _initialize_events(t0);
    JacMat<T, N> tmp(this->Nsys(), this->Nsys());
    this->_rhs(tmp.data(), t0, q0.data());
    if (!all_are_finite(tmp.data(), this->Nsys())){
        this->kill("Initial ode rhs is nan or inf");
    }
    if (_ode.jacobian != nullptr && (IS_IMPLICIT)){
        this->_jac(tmp, t0, q0);
        if (!all_are_finite(tmp.data(), tmp.size())){
            this->kill("Initial Jacobian is nan or inf");
        }
    }
}


template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::t() const {
    return (_event_idx == -1) ? current_state().t : _events.state(_event_idx).t;
}

template<typename T, size_t N, typename Derived>
inline const Array1D<T, N>& DerivedSolver<T, N, Derived>::q() const {
    return (_event_idx == -1) ? current_state().vector : _events.state(_event_idx).exp_vec();
}

template<typename T, size_t N, typename Derived>
inline const Array1D<T, N>& DerivedSolver<T, N, Derived>::true_vector() const {
    return (_event_idx == -1) ? current_state().vector : _events.state(_event_idx).true_vec();
}

template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::stepsize() const {
    return current_state().habs;
}

template<typename T, size_t N, typename Derived>
inline int DerivedSolver<T, N, Derived>::direction() const {
    return _direction;
}

template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::rtol() const {
    return _rtol;
}

template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::atol() const {
    return _atol;
}

template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::min_step() const {
    return _min_step;
}

template<typename T, size_t N, typename Derived>
inline const T& DerivedSolver<T, N, Derived>::max_step() const {
    return _max_step;
}

template<typename T, size_t N, typename Derived>
inline const std::vector<T>& DerivedSolver<T, N, Derived>::args() const {
    return _args;
}

template<typename T, size_t N, typename Derived>
inline size_t DerivedSolver<T, N, Derived>::Nsys() const {
    return _n;
}

template<typename T, size_t N, typename Derived>
inline size_t DerivedSolver<T, N, Derived>::Nupdates() const {
    return _N;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::diverges() const {
    return _diverges;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::is_running() const {
    return _is_running;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::is_dead() const {
    return _is_dead;
}

template<typename T, size_t N, typename Derived>
inline std::string DerivedSolver<T, N, Derived>::message() const {
    return _message;
}

template<typename T, size_t N, typename Derived>
inline SolverState<T, N> DerivedSolver<T, N, Derived>::state() const {
    return SolverState<T, N>({this->t(), this->q(), this->stepsize()}, this->current_events(), this->diverges(), this->is_running(), this->is_dead(), this->Nupdates(), this->message());
}

template<typename T, size_t N, typename Derived>
inline std::vector<const Event<T, N>*> DerivedSolver<T, N, Derived>::current_events() const {
    std::vector<const Event<T, N>*> events(0);
    for (const size_t* i=_events.begin(); i != _events.end(); ++i){
        events.push_back(&_events.event(*i));
    }
    return events;
}

template<typename T, size_t N, typename Derived>
inline const EventCollection<T, N>& DerivedSolver<T, N, Derived>::event_col() const {
    return _events;
}

template<typename T, size_t N, typename Derived>
inline std::string DerivedSolver<T, N, Derived>::name() const {
    return _name;
}


template<typename T, size_t N, typename Derived>
T DerivedSolver<T, N, Derived>::auto_step(const ICS<T, N>* ics) const {
    //returns absolute value of emperically determined first step.
    const int dir = _direction;
    const T& t = (ics == nullptr) ? current_state().t : ics->t;
    const Array1D<T, N>& q = (ics == nullptr) ? current_state().vector : ics->q;

    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    size_t n = this->Nsys();
    T h0, d2, h1;
    Array1D<T, N> y1(n), f1(n);
    Array1D<T, N> scale(n);
    for (size_t i=0; i<n; i++){
        scale[i] = _atol + abs(q[i])*_rtol;
    }
    Array1D<T, N> f0(n);
    this->_rhs(f0.data(), t, q.data());
    T d0 = rms_norm(q.data(), scale.data(), n);
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
    _rhs(f1.data(), t+h0*dir, y1.data());
    Array1D<T, N> tmp(n);
    for (size_t i=0; i<n; i++){
        tmp[i] = f1[i] - f0[i];
    }
    d2 = rms_norm(tmp.data(), scale.data(), n) / h0;
    
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = std::max(T(1)/1000000, h0/1000);
    }
    else{
        h1 = pow(100*std::max(d1, d2), -T(1)/T(Derived::ERR_EST_ORDER+1));
    }
    return std::max(std::min({100*h0, h1, this->_max_step}), this->_min_step);
}


template<typename T, size_t N, typename Derived>
inline OdeSolver<T, N>* DerivedSolver<T, N, Derived>::clone() const{
    return new Derived(*THIS_C);
}

template<typename T, size_t N, typename Derived>
inline const Interpolator<T, N>* DerivedSolver<T, N, Derived>::interpolator() const{
    return &_current_linked_interpolator;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::is_interpolating() const{
    return _interp_data;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::at_event() const{
    return _event_idx != -1;
}

template<typename T, size_t N, typename Derived>
inline const State<T, N>& DerivedSolver<T, N, Derived>::ics() const{
    return _states[0];
}


template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::advance(){
    if (this->_is_dead){
        this->_warn_dead();
        return false;
    }
    else if (!this->_is_running){
        this->_warn_paused();
        return false;
    }

    if (this->_requires_new_start()){
        _states[_curr_state_idx] = {this->t(), this->true_vector(), this->stepsize()};
        re_adjust();
    }

    if (_equiv_states()){
        this->adapt_impl(_states[_aux_state_idx]);
        if (_validate_it(_states[_aux_state_idx])){
            _register_states();
            _events.detect_all_between(this->old_state(), this->current_state(), interp_func<T, N, Derived>, this);
            if (_interp_data){
                std::unique_ptr<Interpolator<T, N>> r = this->state_interpolator(0, -1);
                if (const EventState<T, N>* ev = _events.canon_state()){
                    r->adjust_end(ev->t);
                }
                this->_add_interpolant(std::move(r));
            }
        }
        else{
            return false;
        }
    }
    else{
        _events.next_result();
    }

    if (_events.begin()){
        _event_idx = *_events.begin();
        if (_interp_data && _requires_new_start()){
            if (!_events.canon_event()->hides_mask()){
                std::unique_ptr<Interpolator<T, N>> r = std::unique_ptr<Interpolator<T, N>>(new LocalInterpolator<T, N>(this->t(), this->true_vector()));
                _current_linked_interpolator.adjust_end(this->t());
                this->_add_interpolant(std::move(r));
            }
        }
    }
    else{
        _event_idx = -1;
    }
    for (size_t idx : _events){
        if (_events.event(idx).is_leathal()){
            this->kill(_events.event(idx).name());
        }
        else if (_events.event(idx).is_stop_event()){
            this->stop(_events.event(idx).name());
        }
    }

    if (_interp_data){
        // _current_linked_interpolator.adjust_end(this->t());
        _current_linked_interpolator.close_end();
    }

    this->_N++;
    return true;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::advance_to_event(){
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


template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::stop(std::string text){
    _is_running = false;
    _message = (text == "") ? "Stopped by user" : text;
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::kill(std::string text){
    _is_running = false;
    _is_dead = true;
    _message = (text == "") ? "Killed by user" : text;
}

template<typename T, size_t N, typename Derived>
bool DerivedSolver<T, N, Derived>::resume(){
    if (_is_dead){
        _warn_dead();
    }
    else{
        _message = "Running";
        _is_running = true;
        return true;
    }
    return false;
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::set_tmax(T tmax){
    _events.set_tmax(tmax);
    this->resume();
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::start_interpolation() {
    if (!_interp_data){
        _interp_data = true;

        LinkedInterpolator<T, N>& cli = _current_linked_interpolator;
        if (this->_equiv_states()){
            cli = LinkedInterpolator<T, N>(this->t(), this->q());
        }
        else{
            int bdr1 = 1;
            if (at_event() && _events.canon_event() && (_events.state(_event_idx).t == _events.canon_state()->t) && _events.canon_event()->hides_mask()){
                cli = LinkedInterpolator<T, N>(this->t(), this->q());
                bdr1 = -1;
            }
            std::unique_ptr<Interpolator<T, N>> r = state_interpolator(bdr1, -1);
            r->adjust_start(this->t());
            
            if (bdr1 == 1){
                cli = LinkedInterpolator<T, N>(r.get());
            }
            else{
                cli.expand_by_owning(std::move(r));
            }

        }
        
    }
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::stop_interpolation() {
    _current_linked_interpolator = LinkedInterpolator<T, N>(this->t(), this->q());
    _interp_data = false;
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::reset() {
    _diverges = false;
    _is_dead = false;
    _N = 0;
    resume();
    _events.reset();
    _event_idx = -1;
    for (size_t i=1; i<_states.size(); i++){
        _states[i] = _states[0];
    }
    _old_state_idx = 1;
    _curr_state_idx = 2;
    _aux_state_idx = 3;
    stop_interpolation();
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::set_obj(const void* obj){
    _ode.obj = obj;
}

template<typename T, size_t N, typename Derived>
inline const State<T, N>& DerivedSolver<T, N, Derived>::old_state() const{
    return _states[_old_state_idx];
}

template<typename T, size_t N, typename Derived>
inline const State<T, N>& DerivedSolver<T, N, Derived>::current_state() const{
    return _states[_curr_state_idx];
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::adapt_impl(State<T, N>& result){
    return THIS->adapt_impl(result);
}

template<typename T, size_t N, typename Derived>
inline std::unique_ptr<Interpolator<T, N>> DerivedSolver<T, N, Derived>::state_interpolator(int bdr1, int bdr2) const{
    return THIS_C->state_interpolator(bdr1, bdr2);
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::_rhs(T* result, const T& t, const T* q) const{
    _ode.rhs(result, t, q, _args.data(), _ode.obj);
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::_jac(JacMat<T, N>& result, const T& t, const Array1D<T, N>& q) const{
    return _ode.jacobian(result.data(), t, q.data(), _args.data(), _ode.obj);
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::interp(T* result, const T& t) const{
    return THIS_C->interp(result, t);
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::re_adjust() {
    THIS->re_adjust();
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::_register_states(){
    size_t tmp = _curr_state_idx;
    _curr_state_idx = _aux_state_idx;
    _aux_state_idx = _old_state_idx;
    _old_state_idx = tmp;
    _event_idx = -1;
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::_initialize_events(const T& t0){
    _events.set_args(this->_args);
    _events.set_start(t0, this->direction());
    _events.set_array_size(this->Nsys());
}

template<typename T, size_t N, typename Derived>
void DerivedSolver<T, N, Derived>::_add_interpolant(std::unique_ptr<Interpolator<T, N>>&& interpolant){
    LinkedInterpolator<T, N>& cli = _current_linked_interpolator;
    if (cli.last_interpolant().interval().is_point() && interpolant->interval().start_bdr() == 0){
        interpolant->close_start();
    }
    cli.expand_by_owning(std::move(interpolant));
}

template<typename T, size_t N, typename Derived>
bool DerivedSolver<T, N, Derived>::_validate_it(const State<T, N>& state){
    bool success = true;
    if (this->is_dead()){
        // The derived adapt_impl may kill the solver under the conditions that it deems so.
        success = false;
    }
    else if (!all_are_finite(state.vector.data(), state.vector.size())){
        this->kill("Ode solution diverges");
        this->_diverges = true;
        success = false;
    }
    else if (state.habs <= MIN_STEP){
        this->kill("Required stepsize was smaller than machine precision");
        success = false;
    }

    return success;
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::_warn_dead() const {
    std::cout << "\n" << "Solver has permanently stop integrating. Termination cause:\n\t" << _message << std::endl;
}

template<typename T, size_t N, typename Derived>
inline void DerivedSolver<T, N, Derived>::_warn_paused() const {
    std::cout << "\n" << "Solver has paused integrating. Please resume the integrator by any means to continue advancing *before* doing so." << std::endl;
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::_requires_new_start() const{
    return _events.canon_event() && (_events.canon_state()->t == this->t());
}

template<typename T, size_t N, typename Derived>
inline bool DerivedSolver<T, N, Derived>::_equiv_states() const{
    return at_event() ? _events.state(_event_idx).t == this->current_state().t : true;
}


template<typename T, size_t N, typename Derived>
inline void interp_func(T* res, const T& t, const void* obj){
    const auto* solver = reinterpret_cast<const DerivedSolver<T, N, Derived>*>(obj);
    solver->interp(res, t);
}


template<typename T, size_t N, typename Derived>
const T DerivedSolver<T, N, Derived>::MAX_FACTOR = T(10);

template<typename T, size_t N, typename Derived>
const T DerivedSolver<T, N, Derived>::SAFETY = T(9)/10;

template<typename T, size_t N, typename Derived>
const T DerivedSolver<T, N, Derived>::MIN_FACTOR = T(2)/10;

template<typename T, size_t N, typename Derived>
const T DerivedSolver<T, N, Derived>::MIN_STEP = 100*std::numeric_limits<T>::epsilon();

#endif