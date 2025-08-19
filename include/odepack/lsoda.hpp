#ifndef LSODA_HPP
#define LSODA_HPP



#include "stiff.hpp"
#include "rk_adaptive.hpp"


template<typename T, int N>
class LSODA final : public OdeSolver<T, N>{

public:

    using UniqueClone = OdeSolver<T, N>::UniqueClone;

    LSODA(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    LSODA(const LSODA& other);

    LSODA(LSODA&& other);

    LSODA& operator=(const LSODA& other);

    LSODA& operator=(LSODA&& other);

    ~LSODA();

    LSODA() = delete;

    const T&                        t() const final;
    const vec<T, N>&                q() const final;
    const vec<T, N>&                q_true() const final;
    const T&                        stepsize() const final;
    const T&                        tmax() const final;
    const int&                      direction() const final;
    const T&                        rtol() const final;
    const T&                        atol() const final;
    const T&                        min_step() const final;
    const T&                        max_step() const final;
    const std::vector<T>&           args() const final;
    const size_t&                   Nsys() const final;
    const size_t&                   Nupdates() const final;
    const bool&                     diverges() const final;
    const bool&                     is_running() const final;
    const bool&                     is_dead() const final;
    const vec<T, N>&                error() const final;
    const std::string&              message() const final;
    const SolverState<T, N>         state() const final;
    const State<T, N>&              full_state() const final;
    const EventCollection<T, N>&    events() const final;
    bool                            at_event() const final;
    std::string                     event_name() const final;
    const Event<T, N>&              current_event() const final;
    const int&                      current_event_index() const final;
    const std::string&              name() const final;
    T                               auto_step(T direction=0, const ICS<T, N>* = nullptr) const final;
    LSODA<T, N>*                    clone() const final;
    UniqueClone                     safe_clone() const final;
    UniqueClone                     with_new_events(const EventCollection<T, N>& events) const final;
    const Interpolator<T, N>*       interpolator() const final;
    inline const bool&              is_interpolating() const final;

    bool                            advance() final;
    bool                            set_goal(const T& t_max) final;
    void                            restart_from(const OdeSolver<T, N>& other) final;
    void                            stop(const std::string& text) final;
    void                            kill(const std::string& text) final;
    bool                            resume() final;
    bool                            free() final;
    void                            start_interpolation() final;
    void                            stop_interpolation() final;

private:

    inline OdeSolver<T, N>& _solver();

    inline const OdeSolver<T, N>& _solver() const;

    inline OdeSolver<T, N>& _other_solver();

    inline const OdeSolver<T, N>& _other_solver() const;

    RK45<T, N> _rk_solver;
    BDF<T, N> _bdf_solver;
    bool _stiff = false;
    size_t _Nupdates = 0;
    vec<T, N> _error;
    std::string _name = "LSODA";
    LinkedInterpolator<T, N> _interpolator;
    bool _interp_is_valid = false;
    mutable _InterpolatorViewer<T, N>* _viewer = nullptr;

};



/*
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----------------------------------------IMPLEMENTATIONS-------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
*/

template<typename T, int N>
LSODA<T, N>::LSODA(MAIN_CONSTRUCTOR(T, N)) : OdeSolver<T, N>(), _rk_solver(ARGS), _bdf_solver(ARGS), _error(q0.size()), _interpolator(t0, q0){
    _error.setZero();
}

template<typename T, int N>
LSODA<T, N>::LSODA(const LSODA& other) : _rk_solver(other._rk_solver), _bdf_solver(other._bdf_solver), _stiff(other._stiff), _Nupdates(other._Nupdates), _error(other._error), _interpolator(other._interpolator), _interp_is_valid(other._interp_is_valid) {}

template<typename T, int N>
LSODA<T, N>::LSODA(LSODA&& other) : _rk_solver(std::move(other._rk_solver)), _bdf_solver(std::move(other._bdf_solver)), _stiff(other._stiff), _Nupdates(other._Nupdates), _error(std::move(other._error)), _interpolator(std::move(other._interpolator)), _interp_is_valid(other._interp_is_valid), _viewer(other._viewer) {
    other._viewer = nullptr;
}

template<typename T, int N>
LSODA<T, N>& LSODA<T, N>::operator=(const LSODA& other){
    if (&other != this){
        OdeSolver<T, N>::operator=(other);
        _rk_solver = other._rk_solver;
        _bdf_solver = other._bdf_solver;
        _stiff = other._stiff;
        _Nupdates = other._Nupdates;
        _error = other._error;
        _interpolator = other._interpolator;
        _interp_is_valid = other._interp_is_valid;
        delete _viewer;
        _viewer = nullptr;
    }
    return *this;
}

template<typename T, int N>
LSODA<T, N>& LSODA<T, N>::operator=(LSODA&& other){
    if (&other != this){
        OdeSolver<T, N>::operator=(std::move(other));
        _rk_solver = std::move(other._rk_solver);
        _bdf_solver = std::move(other._bdf_solver);
        _stiff = other._stiff;
        _Nupdates = other._Nupdates;
        _error = std::move(other._error);
        _interpolator = std::move(other._interpolator);
        _interp_is_valid = other._interp_is_valid;
        delete _viewer;
        _viewer = other._viewer;
        other._viewer = nullptr;
    }
    return *this;
}

template<typename T, int N>
LSODA<T, N>::~LSODA(){
    delete _viewer;
    _viewer = nullptr;
}


template<typename T, int N>
inline OdeSolver<T, N>& LSODA<T, N>::_solver(){
    return _stiff ? static_cast<OdeSolver<T, N>&>(_bdf_solver) : static_cast<OdeSolver<T, N>&>(_rk_solver);
}

template<typename T, int N>
inline const OdeSolver<T, N>& LSODA<T, N>::_solver() const {
    return _stiff ? static_cast<const OdeSolver<T, N>&>(_bdf_solver) : static_cast<const OdeSolver<T, N>&>(_rk_solver);
}

template<typename T, int N>
inline OdeSolver<T, N>& LSODA<T, N>::_other_solver(){
    return !_stiff ? static_cast<OdeSolver<T, N>&>(_bdf_solver) : static_cast<OdeSolver<T, N>&>(_rk_solver);
}

template<typename T, int N>
inline const OdeSolver<T, N>& LSODA<T, N>::_other_solver() const {
    return !_stiff ? static_cast<const OdeSolver<T, N>&>(_bdf_solver) : static_cast<const OdeSolver<T, N>&>(_rk_solver);
}

template<typename T, int N>
const T& LSODA<T, N>::t() const {
    return _solver().t();
}

template<typename T, int N>
const vec<T, N>& LSODA<T, N>::q() const {
    return _solver().q();
}

template<typename T, int N>
const vec<T, N>& LSODA<T, N>::q_true() const {
    return _solver().q_true();
}

template<typename T, int N>
const T& LSODA<T, N>::stepsize() const {
    return _solver().stepsize();
}

template<typename T, int N>
const T& LSODA<T, N>::tmax() const {
    return _solver().tmax();
}

template<typename T, int N>
const int& LSODA<T, N>::direction() const {
    return _solver().direction();
}

template<typename T, int N>
const T& LSODA<T, N>::rtol() const {
    return _solver().rtol();
}

template<typename T, int N>
const T& LSODA<T, N>::atol() const {
    return _solver().atol();
}

template<typename T, int N>
const T& LSODA<T, N>::min_step() const {
    return _solver().min_step();
}

template<typename T, int N>
const T& LSODA<T, N>::max_step() const {
    return _solver().max_step();
}

template<typename T, int N>
const std::vector<T>& LSODA<T, N>::args() const {
    return _solver().args();
}

template<typename T, int N>
const size_t& LSODA<T, N>::Nsys() const {
    return _solver().Nsys();
}

template<typename T, int N>
const size_t& LSODA<T, N>::Nupdates() const {
    return _Nupdates;
}

template<typename T, int N>
const bool& LSODA<T, N>::diverges() const {
    return _solver().diverges();
}

template<typename T, int N>
const bool& LSODA<T, N>::is_running() const {
    return _solver().is_running();
}

template<typename T, int N>
const bool& LSODA<T, N>::is_dead() const {
    return _solver().is_dead();
}

template<typename T, int N>
const vec<T, N>& LSODA<T, N>::error() const {
    return _error;
}

template<typename T, int N>
const std::string& LSODA<T, N>::message() const {
    return _solver().message();
}

template<typename T, int N>
const SolverState<T, N> LSODA<T, N>::state() const {
    return _solver().state();
}

template<typename T, int N>
const State<T, N>& LSODA<T, N>::full_state() const {
    return _solver().full_state();
}

template<typename T, int N>
const EventCollection<T, N>& LSODA<T, N>::events() const {
    return _solver().events();
}

template<typename T, int N>
bool LSODA<T, N>::at_event() const {
    return _solver().at_event();
}

template<typename T, int N>
std::string LSODA<T, N>::event_name() const {
    return _solver().event_name();
}

template<typename T, int N>
const Event<T, N>& LSODA<T, N>::current_event() const {
    return _solver().current_event();
}

template<typename T, int N>
const int& LSODA<T, N>::current_event_index() const {
    return _solver().current_event_index();
}

template<typename T, int N>
const std::string& LSODA<T, N>::name() const {
    return _name;
}

template<typename T, int N>
T LSODA<T, N>::auto_step(T direction, const ICS<T, N>* ics) const {
    return _solver().auto_step(direction, ics);
}

template<typename T, int N>
LSODA<T, N>* LSODA<T, N>::clone() const {
    return new LSODA<T, N>(*this);
}

template<typename T, int N>
LSODA<T, N>::UniqueClone LSODA<T, N>::safe_clone() const {
    return std::make_unique<LSODA<T, N>>(*this);
}

template<typename T, int N>
LSODA<T, N>::UniqueClone LSODA<T, N>::with_new_events(const EventCollection<T, N>& events) const {
    LSODA<T, N>* ptr = this->clone();
    ptr->_rk_solver = static_cast<const RK45<T, N>&>(*this->_rk_solver.with_new_events(events));
    ptr->_bdf_solver = static_cast<const BDF<T, N>&>(*this->_bdf_solver.with_new_events(events));
    return std::unique_ptr<OdeSolver<T, N>>(ptr);
}

template<typename T, int N>
const Interpolator<T, N>* LSODA<T, N>::interpolator() const {

    if (is_interpolating()){
        if (_interp_is_valid){
            if (_viewer == nullptr){
                _viewer = new _InterpolatorViewer<T, N>({&_interpolator, _solver().interpolator()});
            }
            else{
                *_viewer = _InterpolatorViewer<T, N>({&_interpolator, _solver().interpolator()});
            }
            return _viewer;
        }
        else{
            return _solver().interpolator();
        }
    }
    else{
        return nullptr;
    }
}

template<typename T, int N>
inline const bool& LSODA<T, N>::is_interpolating() const {
    return _solver().is_interpolating();
}


template<typename T, int N>
bool LSODA<T, N>::advance() {
    if (_solver().advance()){
        _Nupdates++;
        _error += _solver().full_state().local_error();

        bool change = false; //TODO: determine automatic stiffness detection algorithm.
        //the entirety of the LSODA algorithm relies on this part of the code.
        //Without this, this header file is useless.

        //TODO. fix bug when interpolating.
        
        if (change){
            _other_solver().restart_from(_solver());

            //now manage interpolators
            if (is_interpolating()){
                if (_interp_is_valid){
                    _interpolator.expand(*_solver().interpolator());
                }
                else{
                    _interpolator = LinkedInterpolator<T, N>(_solver().interpolator());
                    _interp_is_valid = true;
                }
                _solver().stop_interpolation();
            }
            _stiff = !_stiff;
        }
        return true;
    }
    return false;
}

template<typename T, int N>
bool LSODA<T, N>::set_goal(const T& t_max_new) {
    int dir = _solver().direction();
    if (_solver().set_goal(t_max_new)){

        if ((_solver().direction() * dir < 0) && is_interpolating()){
            stop_interpolation();
        }

        return true;
    }
    return false;
}

template<typename T, int N>
void LSODA<T, N>::restart_from(const OdeSolver<T, N>& other){
    if (_bdf_solver.is_interpolating()){
        _bdf_solver.stop_interpolation();
    }
    _stiff = false;
    _rk_solver.restart_from(other);
    _Nupdates = 0;
    _error.setZero();
    _interpolator = LinkedInterpolator<T, N>(this->t(), this->q_true());
    _interp_is_valid = false;
}

template<typename T, int N>
void LSODA<T, N>::stop(const std::string& text) {
    _solver().stop(text);
}

template<typename T, int N>
void LSODA<T, N>::kill(const std::string& text) {
    _solver().kill(text);
}

template<typename T, int N>
bool LSODA<T, N>::resume() {
    return _solver().resume();
}

template<typename T, int N>
bool LSODA<T, N>::free() {
    return _solver().free();
}

template<typename T, int N>
void LSODA<T, N>::start_interpolation() {
    _solver().start_interpolation();
}

template<typename T, int N>
void LSODA<T, N>::stop_interpolation() {
    _solver().stop_interpolation();
    _interpolator = LinkedInterpolator<T, N>(this->t(), this->q());
    _interp_is_valid = false;
}


#endif