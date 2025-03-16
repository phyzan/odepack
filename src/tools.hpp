#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <map>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <fstream>


// USEFUL ALIASES

template<class Tt, int N=-1>
using vec = Eigen::Array<Tt, 1, N>;

template<class Tt, class Ty>
using Func = std::function<Ty(const Tt&, const Ty&, const std::vector<Tt>&)>;

template<class Tt, class Ty>
using event_f = std::function<Tt(const Tt&, const Ty&, const std::vector<Tt>&)>;

template<class Tt, class Ty>
using is_event_f = std::function<bool(const Tt&, const Ty&, const std::vector<Tt>&)>;

template<class Tt>
using _ObjFun = std::function<Tt(const Tt&)>;

using _Shape = std::vector<size_t>;

template<class T>
using complex = std::complex<T>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag;


template<class T, int Nr, int Nc>
T norm(const Eigen::Array<T, Nr, Nc>& f){
    return (f*f).sum();
}

template<class T, int Nr, int Nc>
Eigen::Array<T, Nr, Nc> cwise_abs(const Eigen::Array<T, Nr, Nc>& f){
    return f.cwiseAbs();
}

template<class T, int Nr, int Nc>
Eigen::Array<T, Nr, Nc> cwise_max(const Eigen::Array<T, Nr, Nc>& a, const Eigen::Array<T, Nr, Nc>& b){
    return a.cwiseMax(b);
}

template<class T>
T abs(const T& x){
    return (x > 0) ? x : -x;
}

template<class T, int Nr, int Nc>
bool All_isFinite(const Eigen::Array<T, Nr, Nc>& arr){
    return arr.isFinite().all();
}

template<class T, int Nr, int Nc>
std::vector<size_t> shape(const Eigen::Array<T, Nr, Nc>& arr){
    return {size_t(arr.rows()), size_t(arr.cols())};
}

template<class T>
std::vector<size_t> shape(const std::vector<T>& arr){
    return {arr.size()};
}


template<class T>
std::vector<T> subvec(const std::vector<T>& x, const size_t& start) {
    if (start >= x.size()) {
        return {}; // Return an empty vector if start is out of bounds
    }
    return std::vector<T>(x.begin() + start, x.end());
}


//BISECTION USED FOR EVENTS IN ODES

template<class T>
std::vector<T> bisect(const _ObjFun<T>& f, const T& a, const T& b, const T& atol){
    T err = 2*atol+1;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    if (f(a)*f(b) > 0){
        throw std::runtime_error("Root not bracketed");
    }

    while (err > atol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            break;
        }
        fm = f(c);
        if (f(_a) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = abs(fm);
    }
    return {_a, c, _b};
}

void _assert_valid_name(const std::string& name){
    if (name == ""){
        throw std::runtime_error("Please provide a non empty name when instanciating an Event-related class");
    }
}



template<class Tt, class Ty>
class Event{

    //_t_event and _q_event implemented as pointers so that they are null when an event has not been determined instead of
    //having a default value, or one determined from a previous .determine() call.

public:
    Event(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr, const Tt& period=0, const Tt& start=0, Func<Tt, Ty> mask=nullptr, const bool& hide_mask=false) : _name(name), _when(when), _check_if(check_if), _period(period), _start(start), _mask(mask), _hide_mask(hide_mask){
        _assert_valid_name(name);
    }

    Event(const Event<Tt, Ty>& other){
        _copy_data(other);
    }

    Event<Tt, Ty>& operator=(const Event<Tt, Ty>& other){
        if (&other == this) return *this;
        _clear();
        _copy_data(other);
        return *this;
    }

    bool determine(const Tt& t1, const Tt& t2, const std::vector<Tt>& args, std::function<Ty(const Tt&)> q, const Tt& tol){
        _clear();
        Tt t_determined = t2;
        bool determined = false;
        if (_when != nullptr){
            _ObjFun<Tt> obj_fun = [this, q, args](const Tt& t) ->Tt {
                return _when(t, q(t), args);
            };

            if (_check_if == nullptr || (_check_if(t1, q(t1), args) && _check_if(t2, q(t2), args))){
                Tt val1 = _when(t1, q(t1), args);
                Tt val2 = _when(t2, q(t2), args);
                if (val1 * val2 <= 0 && val1 != 0){
                    t_determined = bisect(obj_fun, t1, t2, tol)[2];
                    determined = true;
                }
            }
        }


        if (_determine_periodic_event_between(t1, t_determined) || determined){
            _realloc();
            _set(t_determined, q(t_determined), args);
            return true;
        }
        else{
            return false;
        }
    }

    const bool& hide_mask()const{
        return _hide_mask;
    }

    bool has_mask()const{
        return _mask != nullptr;
    }

    void go_back(){
        _clear();
        if (_is_periodic_event){
            _np = _np_previous;
        }
        _is_periodic_event = false;
    }

    const Tt& t_event()const{
        return *_t_event;
    }

    const Ty& q_event()const{
        return *_q_event;
    }

    const Ty& q_masked()const{
        return *_q_masked;
    }

    const std::string& name()const{
        return _name;
    }

    ~Event(){
        _clear();
    }


private:

    std::string _name;
    event_f<Tt, Ty> _when;
    is_event_f<Tt, Ty> _check_if = nullptr;
    Tt _period;
    Tt _start;
    long int _np = 0;
    long int _np_previous = 0;
    bool _is_periodic_event = false;
    Func<Tt, Ty> _mask = nullptr;
    bool _hide_mask; // variable that is only used externally for odesolvers to determine when and whether to call q_event or q_masked.
    Tt* _t_event = nullptr;
    Ty* _q_event = nullptr;
    Ty* _q_masked = nullptr;

    void _copy_data(const Event<Tt, Ty>& other){
        _name = other._name;
        _when = other._when;
        _check_if = other._check_if;
        _period = other._period;
        _start = other._start;
        _mask = other._mask;
        _hide_mask = other._hide_mask;
        _np = other._np;
        _np_previous = other._np_previous;
        _is_periodic_event = other._is_periodic_event;
        _clear();
        if (other._t_event != nullptr){
            _realloc();
            *_t_event = *other._t_event;
            *_q_event = *other._q_event;
            *_q_masked = *other._q_masked; //this line has the same effect whether _mask is null or not (which is whether _q_masked != _q_event or not)
        }
    }

    void _clear(){
        delete _t_event;
        delete _q_event;
        if (_q_masked != _q_event){
            delete _q_masked;
        }
        _t_event = nullptr;
        _q_event = nullptr;
        _q_masked = nullptr;
    }

    void _realloc(){
        //always used after _clear();
        _t_event = new Tt;
        _q_event = new Ty;
        if (_mask != nullptr){
            _q_masked = new Ty;
        }
        else{
            _q_masked = _q_event;
        }

    }

    bool _determine_periodic_event_between(const Tt& t1, Tt& t2) {
        //if between t1 and t2, return it, else return t2 ((*_t_event)*dir > dir*_t_period+_period)
        if (_period <= 0){
            return false;
        }

        const int direction = (t2 > t1) ? 1 : -1;
        const Tt next = _start+(_np+direction)*_period;
        if ( (t2*direction >= next*direction) && (next*direction > t1*direction) ){
            _np_previous = _np;
            _np += direction;
            t2 = next;
            _is_periodic_event = true;
            return true;
        }
        else{
            _is_periodic_event = false;
            return false;
        }
    }

    void _set(const Tt& t, const Ty& q, const std::vector<Tt>& args){
        //always called right after _realloc(), before calling _clear() e.g.;
        *_t_event = t;
        *_q_event = q;
        if (_mask != nullptr){
            //this also means that _q_masked already points to a different memory location from _realloc();
            *_q_masked = _mask(*_t_event, q, args);
        }
    }
};





template<class Tt, class Ty>
class StopEvent{

public:
    StopEvent(const std::string& name, event_f<Tt, Ty> when, is_event_f<Tt, Ty> check_if=nullptr) : _name(name), _when(when), _check_if(check_if){
        _assert_valid_name(name);
    }

    bool is_between(const Tt& t1, const Ty& q1, const Tt& t2, const Ty& q2, const std::vector<Tt>& args)const{
        if (_check_if == nullptr || (_check_if(t1, q1, args) && _check_if(t2, q2, args))){
            return _when(t1, q1, args) * _when(t2, q2, args) <= 0;
        }
        else{
            return false;
        }
    }

    const std::string& name()const{
        return _name;
    }


private:

    std::string _name;
    event_f<Tt, Ty> _when = nullptr;
    is_event_f<Tt, Ty> _check_if = nullptr;

};


//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<class Tt, class Ty>
struct OdeResult{


    const std::vector<Tt> t;
    const std::vector<Ty> q;
    const std::map<std::string, std::vector<size_t>> events;
    const bool diverges;
    const bool is_stiff;
    const bool success;// if the OdeSolver didnt die during the integration
    const double runtime;
    const std::string message;

    void examine() const{
        std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n" <<
        "\tPoints           : " << t.size() << "\n" <<
        "\tDiverges         : " << (diverges ? "true" : "false") << "\n" << 
        "\tStiff            : " << (is_stiff ? "true" : "false") << "\n" <<
        "\tSuccess          : " << (success ? "true" : "false") << "\n" <<
        "\tRuntime          : " << runtime << "\n" <<
        "\tTermination cause: " << message << "\n" <<
        event_log();
    }

    std::string event_log() const{
        std::string res = "";
        res += "\tEvents:\n\t----------\n";
        for (const auto& [name, array] : events){
            res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
        }
        res += "\n\t----------\n";
        return res;
    }
    
};


template<class Tt, class Ty>
class SolverState{

public:
    const Tt t;
    const Ty q;
    const Tt habs;
    const std::string event;
    const bool diverges;
    const bool is_stiff;
    const bool is_running; //if tmax or breakcond are met or is dead, it is set to false. It can be set to true if new tmax goal is set
    const bool is_dead; //e.g. if stiff or diverges. This is irreversible.
    const size_t N;
    const std::string message;

    SolverState(const Tt& t, const Ty& q, const Tt& habs, const std::string& event, const bool& diverges, const bool& is_stiff, const bool& is_running, const bool& is_dead, const size_t& N, const std::string& message): t(t), q(q), habs(habs), event(event), diverges(diverges), is_stiff(is_stiff), is_running(is_running), is_dead(is_dead), N(N), message(message) {}

    void show(const int& precision = 15) const{

        std::cout << std::endl << std::setprecision(precision) << 
        "OdeSolver current state:\n---------------------------\n"
        "\tt          : " << t << "\n" <<
        "\tq          : " << q << "\n" <<
        "\th          : " << habs << "\n\n";
        std::cout << ((event == "") ? "\tNo event" : "\tEvent      : " + (event) )<< "\n" <<
        "\tDiverges   : " << (diverges ? "true" : "false") << "\n" << 
        "\tStiff      : " << (is_stiff ? "true" : "false") << "\n" <<
        "\tRunning    : " << (is_running ? "true" : "false") << "\n" <<
        "\tUpdates    : " << N << "\n" <<
        "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
        "\tState      : " << message << "\n";
    }




};


template<class Tt, class Ty>
struct State{

    Tt t;
    Ty q;
    Tt h_next;
};


template<class Tt, class Ty>
struct SolverArgs{

    const Func<Tt, Ty> f;
    const Tt t0;
    const Tt tmax;
    const Ty q0;
    const Tt habs;
    const Tt rtol;
    const Tt atol;
    const Tt h_min;
    const std::vector<Tt> args;
    const std::vector<Event<Tt, Ty>> events;
    const std::vector<StopEvent<Tt, Ty>> stop_events;
    const Tt event_tol;
    const std::string save_dir;
    const bool save_events_only;
};


template<class Tt, class Ty>
void write_chechpoint(std::ofstream& file, const Tt& t, const Ty& q, const int& event_index){
    file << event_index << " " << t;
    for (size_t i=0; i<static_cast<size_t>(q.size()); i++){
        file << " " << q[i];
    }
    file << "\n";
}








#endif