#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP


#include "events.hpp"
#include "ode.hpp"
#include "virtualsolver.hpp"
#include <memory>
#include <stdexcept>


// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
void normalize(T* res, const T&, const T* q, const T*, const void* obj);

template<typename T>
void normalized(T* q, size_t nsys);


template<typename T, size_t N>
class NormalizationEvent: public ObjectOwningEvent<PeriodicEvent<T, N>, size_t>{

    using Base = ObjectOwningEvent<PeriodicEvent<T, N>, size_t>;

public:

    NormalizationEvent(const std::string& name, size_t Nsys, const T& period, const T& start = inf<T>());

    DEFAULT_RULE_OF_FOUR(NormalizationEvent);

    void go_back() override;

    const T& logksi() const;

    T lyap() const;

    T t_lyap() const;

    T delta_s() const;

    NormalizationEvent<T, N>* clone() const override;

    void set_start(const T& t) override;

    inline size_t Nsys() const;

    void reset() override;

private:

    void _register_it(const EventState<T, N>& res, State<const T> before, State<const T> after) override;

    T _t_renorm;
    T _t_last;

    T _logksi=0;
    T _logksi_last=0;

    T _delta_s = 0;
    T _delta_s_last = 0;

    int _dir = 0;
    int _dir_last = 0;
};

template<typename T, size_t N>
class VariationalODE : public ODE<T, N>{

public:

    VariationalODE(OdeData<T> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T first_step=0, int dir=1, const std::vector<T>& args = {}, const std::vector<const Event<T, N>*>& events = {}, const std::string& method = "RK45");

    DEFAULT_RULE_OF_FOUR(VariationalODE);

    ODE<T, N>* clone() const override;

    inline const std::vector<T>& t_lyap() const;

    inline const std::vector<T>& lyap() const;

    inline const std::vector<T>& kicks() const;

    void clear() override;

    void reset() override;

private:

    inline const NormalizationEvent<T, N>& _main_event()const;

    void _register_event(size_t event) override;

    size_t _ind = 1;
    std::vector<T> _t_lyap = {};
    std::vector<T> _lyap_array = {};
    std::vector<T> _delta_s_arr = {};

};


// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

template<typename T>
void normalize(T* res, const T&, const T* q, const T*, const void* obj){
    const size_t& n = *reinterpret_cast<const size_t*>(obj);
    T N = norm(q+n, n);
    for (size_t i=0; i<n; i++){
        res[i] = q[i];
        res[i+n] = q[i+n]/N;
    }
}

template<typename T>
void normalized(T* q, size_t nsys){
    T N = norm(q+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        q[i+nsys] /= N;
    }
}

// NormalizationEvent implementations
template<typename T, size_t N>
NormalizationEvent<T, N>::NormalizationEvent(const std::string& name, size_t Nsys, const T& period, const T& start): Base(Nsys, name, period, start, normalize<T>, true, nullptr), _t_renorm(start), _t_last(start){}

template<typename T, size_t N>
void NormalizationEvent<T, N>::go_back(){
    PeriodicEvent<T, N>::go_back();
    _t_renorm = _t_last;
    _logksi = _logksi_last;
    _delta_s = _delta_s_last;
    _dir = _dir_last;
}

template<typename T, size_t N>
const T& NormalizationEvent<T, N>::logksi() const{
    return _logksi;
}

template<typename T, size_t N>
T NormalizationEvent<T, N>::lyap() const{
    return _logksi/this->t_lyap();
}

template<typename T, size_t N>
T NormalizationEvent<T, N>::t_lyap() const{
    return abs(_t_renorm-this->_start+this->_period*_dir);
}

template<typename T, size_t N>
T NormalizationEvent<T, N>::delta_s() const{
    return _delta_s;
}

template<typename T, size_t N>
NormalizationEvent<T, N>* NormalizationEvent<T, N>::clone() const{
    return new NormalizationEvent<T, N>(*this);
}

template<typename T, size_t N>
void NormalizationEvent<T, N>::set_start(const T& t){
    PeriodicEvent<T, N>::set_start(t);
    _t_renorm = t;
}

template<typename T, size_t N>
inline size_t NormalizationEvent<T, N>::Nsys() const{
    return this->_object;
}

template<typename T, size_t N>
void NormalizationEvent<T, N>::reset(){
    PeriodicEvent<T, N>::reset();
    _t_renorm = this->_start;
    _t_last = this->_start;
    _logksi = 0;
    _logksi_last = 0;
    _delta_s = 0;
    _delta_s_last = 0;
    _dir = 0;
    _dir_last = 0;
}

template<typename T, size_t N>
void NormalizationEvent<T, N>::_register_it(const EventState<T, N>& res, State<const T> before, State<const T> after){
    _t_last = _t_renorm;
    _logksi_last = _logksi;
    _dir_last = _dir;
    _t_renorm = res.t();
    _delta_s_last = _delta_s;
    _delta_s = log(norm(res.exposed().vector()+Nsys(), Nsys()));
    _logksi += _delta_s;
    _dir = sgn(before.t(), after.t());
}

// VariationalODE implementations
template<typename T, size_t N>
VariationalODE<T, N>::VariationalODE(OdeData<T> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const std::vector<T>& args, const std::vector<const Event<T, N>*>& events, const std::string& method) : ODE<T, N>(){
    for (size_t i=0; i<events.size(); i++){
        if (dynamic_cast<const NormalizationEvent<T, N>*>(events[i])){
            throw std::runtime_error("Initializing a VariationalOdeSolver requires that no normalization events are passed in the constructor");
        }
    }

    Array1D<T, N, Allocation::Auto> tmp(q0_, 2*nsys);
    normalized<T>(tmp.data(), nsys);
    NormalizationEvent<T, N> extra_event("Normalization", nsys, period);
    events.insert(events.begin(), &extra_event);
    nsys *= 2;
    const T* q0 = tmp.data();
    this->_init(ARGS, events, method);
}

template<typename T, size_t N>
ODE<T, N>* VariationalODE<T, N>::clone() const{
    return new VariationalODE<T, N>(*this);
}

template<typename T, size_t N>
inline const std::vector<T>& VariationalODE<T, N>::t_lyap() const{
    return _t_lyap;
}

template<typename T, size_t N>
inline const std::vector<T>& VariationalODE<T, N>::lyap() const{
    return _lyap_array;
}

template<typename T, size_t N>
inline const std::vector<T>& VariationalODE<T, N>::kicks() const{
    return _delta_s_arr;
}

template<typename T, size_t N>
void VariationalODE<T, N>::clear(){
    ODE<T, N>::clear();
    _t_lyap.clear();
    _t_lyap.shrink_to_fit();
    _lyap_array.clear();
    _lyap_array.shrink_to_fit();
    _delta_s_arr.clear();
    _delta_s_arr.shrink_to_fit();
}

template<typename T, size_t N>
void VariationalODE<T, N>::reset(){
    ODE<T, N>::reset();
    _t_lyap.clear();
    _t_lyap.shrink_to_fit();
    _lyap_array.clear();
    _lyap_array.shrink_to_fit();
    _delta_s_arr.clear();
    _delta_s_arr.shrink_to_fit();
}

template<typename T, size_t N>
inline const NormalizationEvent<T, N>& VariationalODE<T, N>::_main_event()const{
    return static_cast<const NormalizationEvent<T, N>&>(this->solver()->event_col().event(_ind));
}

template<typename T, size_t N>
void VariationalODE<T, N>::_register_event(size_t event){
    ODE<T, N>::_register_event(event);
    if (event == _ind-1){
        const NormalizationEvent<T, N>& ev = _main_event();
        _t_lyap.push_back(ev.t_lyap());
        _lyap_array.push_back(ev.lyap());
        _delta_s_arr.push_back(ev.delta_s());
    }
}


#endif
