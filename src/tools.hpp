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

template<class T, int N>
using Fvoidptr = void(*)(vec<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&);

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


template<class T>
void mat_vec_prod(T* result, const T* mat, const T* vec, const size_t& rows, const size_t& cols, const T& factor=1){
    /*
    result[i] = sum_j mat[i, j] * vec[j]
    */
    for (size_t i=0; i<rows; i++){
        T _sum = 0;
        for (size_t j=0; j<cols; j++){
            _sum += mat[i*cols+j]*vec[j];
        }
        result[i] = _sum*factor;
    }
}

template<class T>
void mat_T_vec_prod(T* result, const T* mat, const T* vec, const size_t& rows, const size_t& cols, const T& factor=1){
    /*
    The same as above but the transpose matrix is used
    */
    for (size_t i=0; i<cols; i++){
        T _sum = 0;
        for (size_t j=0; j<rows; j++){
            _sum += mat[j*cols+i]*vec[j];
        }
        result[i] = _sum*factor;
    }
}

template<class S>
void mat_mat_prod(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n, const S& factor=1){
    /*
    a : (m x s)
    b : (s x n)
    */
    for (size_t k=0; k<m*n; k++){
        size_t i = k/n;
        size_t j = k % n;
        S _sum = 0;
        for (size_t q=0; q<s; q++){
            _sum += a[i*s + q] * b[q*n + j];
        }
        r[i*n+j] = _sum*factor;
    }
}


template<class S>
void mat_T_mat_prod(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n, const S& factor=1){
    /*
    a : (s x m)
    b : (s x n)
    */
    for (size_t k=0; k<m*n; k++){
        size_t i = k/n;
        size_t j = k % n;
        S _sum = 0;
        for (size_t q=0; q<s; q++){
            _sum += a[q*n+j] * b[q*n + j];
        }
        r[i*n+j] = _sum*factor;
    }
}


template<class T, int N>
struct Functor{

    using Fvoid = std::function<void(vec<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&)>;

    Functor(){}

    Functor(std::nullptr_t ptr) : func(nullptr){}

    Functor(const Fvoid& f):func(f){}

    Functor(const Func<T, N>& f): func([f](vec<T, N>& res, const T& t, const vec<T, N>& q, const std::vector<T>& args){res = f(t, q, args); }){}

    Functor(const Fvoidptr<T, N>& f):func(f){}

    Functor(const Fptr<T, N>& f): Functor([f](vec<T, N>& res, const T& t, const vec<T, N>& q, const std::vector<T>& args){res = f(t, q, args); }){}

    inline void operator()(vec<T, N>& result, const T& t, const vec<T, N>& q, const std::vector<T>& args)const{
        func(result, t, q, args);
    }

    inline vec<T, N> operator()(const T& t, const vec<T, N>& q, const std::vector<T>& args)const{
        vec<T, N> res(q.size());
        func(res, t, q, args);
        return res;
    }

    Functor<T, N>& operator=(const Fvoid& new_func){
        func = new_func;
        return *this;
    }

    Functor<T, N>& operator=(const Func<T, N>& f){
        func = [f](vec<T, N>& res, const T& t, const vec<T, N>& q, const std::vector<T>& args){res = f(t, q, args); };
        return *this;
    }

    bool operator==(const Functor<T, N>& other){
        return (this == &other) ? true : other.func == func;
    }

    template<class Any>
    bool operator==(const Any& other){
        return func == other;
    }

    Fvoid func=nullptr;
};


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