#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP


#include "events.hpp"
#include "ode.hpp"
#include "odesolvers.hpp"
#include <memory>
#include <stdexcept>

template<typename T>
void normalize(T* res, const T&, const T* q, const T*, const void* obj){
    const size_t& n = *reinterpret_cast<const size_t*>(obj);
    T N = norm(q+n, n);
    for (size_t i=0; i<n; i++){
        res[i] = q[i];
        res[i+n] = q[i+n]/N;
    }
}

template<typename T, size_t N>
Array1D<T, N> normalized(const Array1D<T, N>& q){
    Array1D<T, N> res(q);
    size_t n = q.size()/2;
    normalize<T>(res.data(), T(0), q.data(), nullptr, &n);
    return res;
}


template<typename T, size_t N>
class NormalizationEvent: public ObjectOwningEvent<PeriodicEvent<T, N>, size_t>{

    using Base = ObjectOwningEvent<PeriodicEvent<T, N>, size_t>;

public:
    
    NormalizationEvent(const std::string& name, size_t Nsys, const T& period, const T& start = inf<T>()): Base(Nsys, name, period, start, normalize<T>, true, nullptr), _t_renorm(start), _t_last(start){}

    DEFAULT_RULE_OF_FOUR(NormalizationEvent);

    void go_back() override{
        PeriodicEvent<T, N>::go_back();
        _t_renorm = _t_last;
        _logksi = _logksi_last;
        _delta_s = _delta_s_last;
        _dir = _dir_last;
    }

    const T& logksi() const{
        return _logksi;
    }

    T lyap() const{
        return _logksi/this->t_lyap();
    }

    T t_lyap() const{
        return abs(_t_renorm-this->_start+this->_period*_dir);
    }

    T delta_s() const{
        return _delta_s;
    }

    NormalizationEvent<T, N>* clone() const override{
        return new NormalizationEvent<T, N>(*this);
    }

    void set_start(const T& t) override{
        PeriodicEvent<T, N>::set_start(t);
        _t_renorm = t;
    }

    size_t Nsys() const{
        return static_cast<size_t>(this->_object);
    }

    void reset() override{
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


private:

    void _register_it(const EventState<T, N>& res, const State<T, N>& before, const State<T, N>& after) override {
        _t_last = _t_renorm;
        _logksi_last = _logksi;
        _dir_last = _dir;
        _t_renorm = res.t;
        _delta_s_last = _delta_s;
        _delta_s = log(norm(res.exp_vec().data()+Nsys(), Nsys()));
        _logksi += _delta_s;
        _dir = sgn(after.t - before.t);
    }
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
    VariationalODE(OdeData<T> ode, const T& t0, Array1D<T, N> q0, const T& period, const T& rtol, const T& atol, const T min_step=0, const T& max_step=inf<T>(), const T first_step=0, int dir=1, const std::vector<T> args = {}, std::vector<const Event<T, N>*> events = {}, const std::string& method = "RK45") : ODE<T, N>(){
        _assert_event(q0);
        for (size_t i=0; i<events.size(); i++){
            if (dynamic_cast<const NormalizationEvent<T, N>*>(events[i])){
                throw std::runtime_error("Initializing a VariationalOdeSolver requires that no normalization events are passed in the constructor");
            }
        }
        q0 = normalized<T, N>(q0);
        size_t Nsys = q0.size()/2;
        NormalizationEvent<T, N> extra_event("Normalization", Nsys, period);
        events.insert(events.begin(), &extra_event);
        this->_init(ARGS, method);
    }

    DEFAULT_RULE_OF_FOUR(VariationalODE);

    ODE<T, N>* clone() const override{
        return new VariationalODE<T, N>(*this);
    }

    inline const std::vector<T>& t_lyap() const{
        return _t_lyap;
    }

    inline const std::vector<T>& lyap() const{
        return _lyap_array;
    }

    inline const std::vector<T>& kicks() const{
        return _delta_s_arr;
    }

    void clear() override{
        ODE<T, N>::clear();
        _t_lyap.clear();
        _t_lyap.shrink_to_fit();
        _lyap_array.clear();
        _lyap_array.shrink_to_fit();
        _delta_s_arr.clear();
        _delta_s_arr.shrink_to_fit();
    }

    void reset() override {
        ODE<T, N>::reset();
        _t_lyap.clear();
        _t_lyap.shrink_to_fit();
        _lyap_array.clear();
        _lyap_array.shrink_to_fit();
        _delta_s_arr.clear();
        _delta_s_arr.shrink_to_fit();
    }


private:

    void _assert_event(const Array1D<T, N>& q0)const{
        if ((q0.size() & 1) != 0){
            throw std::runtime_error("Variational ODEs require an even number of system size");
        }
    }

    inline const NormalizationEvent<T, N>& _main_event()const{
        return static_cast<const NormalizationEvent<T, N>&>(this->solver()->event_col().event(_ind));
    }

    void _register_event(size_t event) override{
        ODE<T, N>::_register_event(event);
        if (event == _ind-1){
            const NormalizationEvent<T, N>& ev = _main_event();
            _t_lyap.push_back(ev.t_lyap());
            _lyap_array.push_back(ev.lyap());
            _delta_s_arr.push_back(ev.delta_s());
        }
    }

    size_t _ind = 1;
    std::vector<T> _t_lyap = {};
    std::vector<T> _lyap_array = {};
    std::vector<T> _delta_s_arr = {};

};

#endif