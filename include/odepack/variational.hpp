#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP


#include "events.hpp"
#include "ode.hpp"
#include "odesolvers.hpp"
#include <memory>
#include <stdexcept>

template<typename T>
void normalize(T* res, const T& t, const T* q, const T* args, const void* obj){
    const size_t& n = *reinterpret_cast<const size_t*>(obj);
    T N = norm(q+n, n);
    for (size_t i=0; i<n; i++){
        res[i] = q[i];
        res[i+n] = q[i+n]/N;
    }
}

template<typename T, int N>
vec<T, N> normalized(const T& t, const vec<T, N>& q, const std::vector<T>& args){
    vec<T, N> res(q);
    size_t n = q.size()/2;
    auto delq = res.segment(n, n);
    delq /= delq.matrix().norm();
    return res;
}

template<typename T, int N>
T delq_norm(const vec<T, N>& q){
    size_t n = q.size()/2;
    auto delq = q.segment(n, n);
    return delq.matrix().norm();
}


template<typename T, int N>
class NormalizationEvent: public ObjectOwningEvent<PeriodicEvent<T, N>, size_t>{

    using Base = ObjectOwningEvent<PeriodicEvent<T, N>, size_t>;

public:
    
    NormalizationEvent(const std::string& name, size_t Nsys, const T& period, const T& start = inf<T>()): Base(Nsys, name, period, start, normalize<T>, true, nullptr), _t_renorm(start), _t_last(start){}

    DEFAULT_RULE_OF_FOUR(NormalizationEvent);

    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, FuncLike<T> q, const void* obj) override{
        if (PeriodicEvent<T, N>::determine(t1, q1, t2, q2, q, obj)){
            _t_last = _t_renorm;
            _logksi_last = _logksi;
            _t_renorm = this->state().t();
            _logksi += std::log(delq_norm(this->state().exposed_vector()));
            return true;
        }
        else{
            return false;
        }
    }

    void go_back() override{
        PeriodicEvent<T, N>::go_back();
        _t_renorm = _t_last;
        _logksi = _logksi_last;
    }

    const T& logksi() const{
        return _logksi;
    }

    T lyap() const{
        return _logksi/(_t_renorm-this->_start);
    }

    T t_lyap() const{
        return _t_renorm-this->_start;
    }

    NormalizationEvent<T, N>* clone() const override{
        return new NormalizationEvent<T, N>(*this);
    }

    void set_start(const T& t) override{
        PeriodicEvent<T, N>::set_start(t);
        _t_renorm = t;
    }

    void reset() override{
        PeriodicEvent<T, N>::reset();
        _t_renorm = this->_start;
        _t_last = this->_start;
        _logksi = 0;
        _logksi_last = 0;
    }


private:
    T _t_renorm;
    T _t_last;

    T _logksi=0;
    T _logksi_last=0;
};


template<typename T, int N>
std::unique_ptr<OdeSolver<T, N>> as_variational(const OdeSolver<T, N>& solver, const T& period){
    NormalizationEvent<T, N> extra_event("Normalization", solver.Nsys()/2, period);
    EventCollection<T, N> events = solver.events().including(&extra_event);
    return solver.with_new_events(events);
}

template<typename T, int N>
class VariationalODE : public ODE<T, N>{


public:
    VariationalODE(OdeData<T> ode, const T& t0, const vec<T, N>& q0, const T& period, const T& rtol, const T& atol, const T min_step=0, const T& max_step=inf<T>(), const T first_step=0, const std::vector<T> args = {}, const EventCollection<T, N>& events = {}, const std::string& method = "RK45") : ODE<T, N>(*as_variational(*get_solver(method, ode, t0, normalized(t0, q0, args), rtol, atol, min_step, max_step, first_step, args, events), period)) {
        _assert_event(this->solver().q());
        _ind = _position_of_main_event();

        for (size_t i=0; i<events.size(); i++){
            if (dynamic_cast<const NormalizationEvent<T, N>*>(&events[i])){
                throw std::runtime_error("Initializing a VariationalOdeSolver requires that no normalization events are passed in the constructor");
            }
        }
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

    void clear() override{
        ODE<T, N>::clear();
        _t_lyap.clear();
        _t_lyap.shrink_to_fit();
        _lyap_array.clear();
        _lyap_array.shrink_to_fit();
    }

    void reset() override {
        ODE<T, N>::reset();
        _t_lyap.clear();
        _t_lyap.shrink_to_fit();
        _lyap_array.clear();
        _lyap_array.shrink_to_fit();
    }


private:

    void _assert_event(const vec<T, N>& q0)const{
        if ((q0.size() & 1) != 0){
            throw std::runtime_error("Variational ODEs require an even number of system size");
        }
    }

    inline const NormalizationEvent<T, N>& _main_event()const{
        return static_cast<const NormalizationEvent<T, N>&>(this->solver().events()[_ind]);
    }

    size_t _position_of_main_event() const{
        for (size_t i=0; i<this->solver().events().size(); i++){
            if (this->solver().events()[i].name() == "Normalization"){
                return i;
            }
        }
        throw std::runtime_error("Normalization event not found");
    }

    void _register_state(const int& event) override{
        ODE<T, N>::_register_state(event);
        if (event == static_cast<int>(_ind)){
            const NormalizationEvent<T, N>& ev = _main_event();
            _t_lyap.push_back(ev.t_lyap());
            _lyap_array.push_back(ev.lyap());
        }
    }

    size_t _ind;
    std::vector<T> _t_lyap = {};
    std::vector<T> _lyap_array = {};

};

#endif