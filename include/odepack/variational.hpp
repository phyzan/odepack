#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP


#include "events.hpp"
#include "ode.hpp"
#include "odesolvers.hpp"
#include <memory>

template<typename T, int N>
vec<T, N> normalize(const T& t, const vec<T, N>& q, const std::vector<T>& args){
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
class NormalizationEvent: public PeriodicEvent<T, N>{

public:
    NormalizationEvent(const std::string& name, const T& period, const T& start): PeriodicEvent<T, N>(name, period, start, normalize<T, N>, true), _t_renorm(start), _t_last(start){}

    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override{
        if (PeriodicEvent<T, N>::determine(t1, q1, t2, q2, q)){
            _t_last = _t_renorm;
            _logksi_last = _logksi;
            _t_renorm = this->state().t();
            _logksi += std::log(delq_norm(this->state().vector()));
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


private:
    T _t_renorm;
    T _t_last;

    T _logksi=0;
    T _logksi_last=0;


};


template<typename T, int N>
std::unique_ptr<OdeSolver<T, N>> as_variational(const OdeSolver<T, N>& solver, const T& period){
    NormalizationEvent<T, N> extra_event("Normalization", period, solver.t()+period);
    EventCollection<T, N> events = solver.events().including(&extra_event);
    return solver.with_new_events(events);
}

template<typename T, int N>
class VariationalODE : public ODE<T, N>{


public:
    VariationalODE(const OdeRhs<T, N>& rhs, const T& t0, const vec<T, N>& q0, const T& period, const T& rtol, const T& atol, const T min_step=0, const T& max_step=inf<T>(), const T first_step=0, const std::vector<T> args = {}, const std::vector<Event<T, N>*>& events = {}, const std::string& method = "RK45") : ODE<T, N>(*as_variational(*get_solver(method, rhs, t0, normalize(t0, q0, args), rtol, atol, min_step, max_step, first_step, args, events), period)) {
        _assert_event(this->solver().q());
        _ind = _position_of_main_event();
    }

    VariationalODE(const VariationalODE& other) = default;

    VariationalODE(VariationalODE&& other) = default;

    VariationalODE<T, N>& operator=(const VariationalODE<T, N>& other) = default;

    VariationalODE<T, N>& operator=(VariationalODE<T, N>&& other) = default;

    ODE<T, N>* clone() const override{
        return new VariationalODE<T, N>(*this);
    }

    OdeResult<T, N> var_integrate(const T& interval, const T& lyap_period, const int& max_prints=0){
        TimePoint t1 = now();
        TimePoint t2;
        const size_t Nt = this->_t_arr.size();
        T t_total = 0;
        T t0 = this->_solver->t();
        while (t_total < interval){
            t_total = std::min(t_total+lyap_period, interval); //we could just add lyap_period, but a roundoff error is slowly added up
            this->go_to(t0+t_total, 0, {{"Normalization", 0}}, max_prints, false);
            if (this->is_dead()){
                break;
            }
            else if (this->_solver->t()==t0+t_total){
                this->_register_lyap();
                if (this->_t_arr.back() != this->_solver->t()){
                    this->_register_state();
                }
            }
        }
        t2 = now();

        OdeResult<T, N> res = {subvec(this->_t_arr, Nt), subvec(this->_q_arr, Nt), this->event_map(Nt), this->_solver->diverges(), !this->_solver->is_dead(), as_duration(t1, t2), this->_solver->message()};
        return res;
    }

    inline const std::vector<T>& t_lyap() const{
        return _t_lyap;
    }

    inline const std::vector<T>& lyap() const{
        return _lyap_array;
    }


private:

    void _assert_event(const vec<T, N>& q0)const{
        if (q0.size()%2 != 0){
            throw std::runtime_error("Variational ODEs require an even number of system size");
        }
    }

    const NormalizationEvent<T, N>& _main_event()const{
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

    void _register_lyap(){
        const NormalizationEvent<T, N>& ev = _main_event();
        if ( (_t_lyap.size() == 0) || (ev.t_lyap() != _t_lyap.back())){
            _t_lyap.push_back(ev.t_lyap());
            _lyap_array.push_back(ev.lyap());
            // if ( (this->_Nevents[_ind].size() == 0) || (this->_Nevents[_ind].back() != this->_t_arr.size()) ){
            //     this->_Nevents[_ind].pushback(_t_arr.size()) //not _t_arr.size() in general
            // }
            
        }
    }

    size_t _ind;
    std::vector<T> _t_lyap = {};
    std::vector<T> _lyap_array = {};

};



template<typename T, int N>
void var_integrate_all(const std::vector<VariationalODE<T, N>*>& list, const T& interval, const T& lyap_period, const int& threads=-1, const bool& display_progress = false){
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    int tot = 0;
    int target = list.size();
    Clock clock;
    clock.start();
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (VariationalODE<T, N>* ode : list){
        ode->var_integrate(interval, lyap_period);
        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;
}



#endif