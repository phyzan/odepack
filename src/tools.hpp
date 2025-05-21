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

template<class T, int N=-1>
using vec = Eigen::Array<T, 1, N>;

template<class T, int N=-1>
using Func = std::function<vec<T, N>(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N>
using Fptr = vec<T, N>(*)(const T&, const vec<T, N>&, const std::vector<T>&);

template<class T>
using _ObjFun = std::function<T(const T&)>;

using _Shape = std::vector<size_t>;

template<class T>
using complex = std::complex<T>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag;

template <typename T>
constexpr T inf() {
    return std::numeric_limits<T>::infinity();
}

template<class T, int Nr, int Nc>
T norm_squared(const Eigen::Array<T, Nr, Nc>& f){
    return (f*f).sum();
}

template<class T, int Nr, int Nc>
T rms_norm(const Eigen::Array<T, Nr, Nc>& f){
    return sqrt(norm_squared(f) / f.size());
}

template<class T, int Nr, int Nc>
Eigen::Array<T, Nr, Nc> cwise_abs(const Eigen::Array<T, Nr, Nc>& f){
    return f.cwiseAbs();
}

template<class T>
T abs(const T& x){
    return (x > 0) ? x : -x;
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

template<class T, int N>
struct OdeResult{


    const std::vector<T> t;
    const std::vector<vec<T, N>> q;
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

    std::vector<T> t_filtered(const std::string& event) const {
        return _event_data(this->t, this->event_map, event);
    }

    std::vector<vec<T, N>> q_filtered(const std::string& event) const {
        return _event_data(this->q, this->event_map, event);
    }
    
};


template<class T, int N>
class SolverState{

public:
    const T t;
    const vec<T, N> q;
    const T habs;
    const std::string event;
    const bool diverges;
    const bool is_stiff;
    const bool is_running; //if tmax or breakcond are met or is dead, it is set to false. It can be set to true if new tmax goal is set
    const bool is_dead; //e.g. if stiff or diverges. This is irreversible.
    const size_t Nt;
    const std::string message;

    SolverState(const T& t, const vec<T, N>& q, const T& habs, const std::string& event, const bool& diverges, const bool& is_stiff, const bool& is_running, const bool& is_dead, const size_t& Nt, const std::string& message): t(t), q(q), habs(habs), event(event), diverges(diverges), is_stiff(is_stiff), is_running(is_running), is_dead(is_dead), Nt(Nt), message(message) {}

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
        "\tUpdates    : " << Nt << "\n" <<
        "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
        "\tState      : " << message << "\n";
    }




};


template<class T, int N>
struct State{

    T t;
    vec<T, N> q;
    T h_next;
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