#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <map>
#include <iomanip>
#include <fstream>
#include "tensors.hpp"


// USEFUL ALIASES

template<class S>
using Jac = std::function<void(Tensor<S>&, const S&, const Tensor<S>&, const std::vector<S>&)>;

template<class S>
using Func = std::function<Ty(const S&, const Tensor<S>&, const std::vector<S>&)>;

template<class S>
using Fptr = Tensor<S>(*)(const S&, const Tensor<S>&, const std::vector<S>&);

template<class S>
using _ObjFun = std::function<S(const S&)>;

template<class T>
using complex = std::complex<T>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag;

template <typename T>
constexpr T inf() {
    return std::numeric_limits<T>::infinity();
}


template<class T>
std::vector<T> subvec(const std::vector<T>& x, const size_t& start) {
    if (start >= x.size()) {
        return {}; // Return an empty vector if start is out of bounds
    }
    return std::vector<T>(x.begin() + start, x.end());
}


template<class T>
std::vector<T> _event_data(const std::vector<T>& q, const std::map<std::string, std::vector<size_t>>& event_map, const std::string& event);

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


//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<class S>
struct OdeResult{


    const std::vector<S> t;
    const std::vector<Tensor<S>> q;
    const std::map<std::string, std::vector<size_t>> event_map;
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
        for (const auto& [name, array] : event_map){
            res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
        }
        res += "\n\t----------\n";
        return res;
    }

    std::vector<S> t_filtered(const std::string& event) const {
        return _event_data(this->t, this->event_map, event);
    }

    std::vector<Tensor<S>> q_filtered(const std::string& event) const {
        return _event_data(this->q, this->event_map, event);
    }
    
};


template<class S>
class SolverState{

using Ty = Tensor<S>;

public:
    const S t;
    const Ty q;
    const S habs;
    const std::string event;
    const bool diverges;
    const bool is_stiff;
    const bool is_running; //if tmax or breakcond are met or is dead, it is set to false. It can be set to true if new tmax goal is set
    const bool is_dead; //e.g. if stiff or diverges. This is irreversible.
    const size_t N;
    const std::string message;

    SolverState(const S& t, const Ty& q, const S& habs, const std::string& event, const bool& diverges, const bool& is_stiff, const bool& is_running, const bool& is_dead, const size_t& N, const std::string& message): t(t), q(q), habs(habs), event(event), diverges(diverges), is_stiff(is_stiff), is_running(is_running), is_dead(is_dead), N(N), message(message) {}

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


template<class S>
struct State{

    S t;
    Tensor<S> q;
    S h_next;
};


template<class T>
std::vector<T> _event_data(const std::vector<T>& q, const std::map<std::string, std::vector<size_t>>& event_map, const std::string& event){
    std::vector<size_t> ind = event_map.at(event);
    std::vector<T> data(ind.size());
    for (size_t i=0; i<data.size(); i++){
        data[i] = q[ind[i]];
    }
    return data;
}


#endif