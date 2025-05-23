#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP


#include "ode.hpp"


template<class T, int N>
struct _TempSolverHolder{

    OdeSolver<T, N>* solver;

    ~_TempSolverHolder(){
        delete solver;
        solver = nullptr;
    }
};

template<class T, int N>
vec<T, N> normalize(const T& t, const vec<T, N>& q, const std::vector<T>& args){
    vec<T, N> res(q);
    size_t n = q.size()/2;
    auto delq = res.segment(n, n);
    delq /= delq.matrix().norm();
    return res;
}

template<class T, int N>
T delq_norm(const vec<T, N>& q){
    size_t n = q.size()/2;
    auto delq = q.segment(n, n);
    return delq.matrix().norm();
}


template<class T, int N>
class NormalizationEvent: public PeriodicEvent<T, N>{

public:
    NormalizationEvent(const std::string& name, const T& period, const T& start): PeriodicEvent<T, N>(name, period, start, normalize<T, N>, true), _t_renorm(start), _t_last(start){}

    NormalizationEvent(const NormalizationEvent<T, N>& other): PeriodicEvent<T, N>(other), _t_renorm(other._t_renorm), _t_last(other._t_last), _logksi(other._logksi), _logksi_last(other._logksi_last) {}


    bool determine(const T& t1, const vec<T, N>& q1, const T& t2, const vec<T, N>& q2, const std::function<vec<T, N>(const T&)>& q) override{
        if (PeriodicEvent<T, N>::determine(t1, q1, t2, q2, q)){
            _t_last = _t_renorm;
            _logksi_last = _logksi;
            _t_renorm = this->data().t();
            _logksi += std::log(delq_norm(this->data().q()));
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



template<class T, int N>
_TempSolverHolder<T, N> variational_solver(const SolverArgs<T, N>& S, const std::string& method, const T& period, const T& start){
    std::vector<Event<T, N>*> new_events(S.events);
    Event<T, N>* event = new NormalizationEvent<T, N>("Normalization", period, start);
    new_events.push_back(event);
    SolverArgs<T, N> new_S(S.jac, S.t0, S.q0, S.rtol, S.atol, S.h_min, S.h_max, S.first_step, S.args, new_events, S.mask, S.save_dir, S.save_events_only);
    OdeSolver<T, N>* solver = getSolver(new_S, method);
    delete event;
    event = nullptr;
    return {solver};
}




template<class T, int N>
class VariationalODE : public ODE<T, N>{


public:
    VariationalODE(const T& period, const T& start, const Jac<T, N> f, const T t0, const vec<T, N> q0, const T rtol, const T atol, const T min_step=0., const T& max_step=inf<T>(), const T first_step=0, const std::vector<T> args = {}, const std::string& method = "RK45", const std::vector<Event<T, N>*>& events = {}, const Func<T, N> mask=nullptr, const std::string& savedir="", const bool& save_events_only=false) : ODE<T, N>(*variational_solver(SolverArgs<T, N>(f, t0, normalize(t0, q0, args), rtol, atol, min_step, max_step, first_step, args, events, mask, savedir, save_events_only), method, period, start).solver){
        _assert_event(q0);
        _ind = _position_of_main_event();
    }

    VariationalODE(const T& period, const T& start, const Func<T, N> f, const T t0, const vec<T, N> q0, const T rtol, const T atol, const T min_step=0., const T& max_step=inf<T>(), const T first_step=0, const std::vector<T> args = {}, const std::string& method = "RK45", const std::vector<Event<T, N>*>& events = {}, const Func<T, N> mask=nullptr, const std::string& savedir="", const bool& save_events_only=false) : ODE<T, N>(*variational_solver(SolverArgs<T, N>(f, t0, normalize(t0, q0, args), rtol, atol, min_step, max_step, first_step, args, events, mask, savedir, save_events_only), method, period, start).solver){
        _assert_event(q0);
        _ind = _position_of_main_event();
    }

    ODE<T, N>* clone() const override{
        return new VariationalODE<T, N>(*this);
    }


    OdeResult<T, N> var_integrate(const T& interval, const T& lyap_period, const int& max_prints=0){
        auto t1 = std::chrono::high_resolution_clock::now();
        const size_t Nt = this->_t_arr.size();
        T t_total = 0;
        T t0 = this->_solver->t();
        while (t_total < interval){
            t_total = std::min(t_total+lyap_period, interval); //we could just add lyap_period, but a roundoff error is slowly added up
            this->go_to(t0+t_total, 0, {{"Normalization", 10}}, max_prints, false);
            if (this->is_dead()){
                break;
            }
            this->_register_lyap();
            if (this->_t_arr.back() != this->_solver->t()){
                this->_register_state();
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
    
        std::chrono::duration<double> rt = t2-t1;

        OdeResult<T, N> res = {subvec(this->_t_arr, Nt), subvec(this->_q_arr, Nt), this->event_map(Nt), this->_solver->diverges(), this->_solver->is_stiff(), !this->_solver->is_dead(), rt.count(), this->_solver->message()};
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
        return *static_cast<const NormalizationEvent<T, N>*>(this->solver().event(_ind));
    }

    size_t _position_of_main_event() const{
        for (size_t i=0; i<this->solver().events_size(); i++){
            if (this->solver().event(i)->name() == "Normalization"){
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



template<class T, int N>
void var_integrate_all(const std::vector<VariationalODE<T, N>*>& list, const T& interval, const T& lyap_period, const int& threads=-1, const int& max_prints = 0){
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (VariationalODE<T, N>* ode : list){
        ode->var_integrate(interval, lyap_period, max_prints);
    }
}



#endif