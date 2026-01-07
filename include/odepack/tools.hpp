#ifndef TOOLS_HPP
#define TOOLS_HPP


#include <complex>
#include <map>
#include <chrono>
#include <omp.h>
#include <cmath>
#include "../ndspan/arrays.hpp"
#ifdef MPREAL
#include "mpreal.h"
#endif


enum class Memory : std::uint8_t { View, Own};

template<typename T, Memory Mem=Memory::View>
class State;


template<typename T>
class State<T, Memory::View>{

    //Does NOT own its data if Memory == Memory::View, only views it.
    //Do not use this object if the underlying data have gove out of scope.

public:

    State(T* data, size_t Nsys) : _data(data), _nsys(Nsys) {}

    const T& t() const{
        return _data[0];
    }

    const T& habs() const{
        return _data[1];
    }

    const T* vector() const{
        return _data+2;
    }
    
    T* vector(){
        return _data+2;
    }

    size_t Nsys() const{
        return _nsys;
    }

private:

    T* _data;
    size_t _nsys;
};


template<typename T>
class State<T, Memory::Own> : private Array1D<T>{

    //Does NOT own its data if Memory == Memory::View, only views it.
    //Do not use this object if the underlying data have gove out of scope.

    using Base = Array1D<T>;

public:

    State(const T* data, size_t Nsys) : Base(data, Nsys+2) {}

    const T& t() const{
        return this->data()[0];
    }

    const T& habs() const{
        return this->data()[1];
    }

    const T* vector() const{
        return this->data()+2;
    }

    T* vector(){
        return this->data()+2;
    }

    size_t Nsys() const{
        return this->size();
    }

protected:

    const T* data() const{
        return Base::data();
    }

    T* data(){
        return Base::data();
    }
};

template<typename T, size_t N>
class EventState{

public:

    EventState() = default;

    const T& t() const{
        return data[0];
    }

    State<const T> True() const{
        return {data.data(), Nsys};
    }

    State<const T> exposed() const{
        return choose_true ? this->True() : State<const T>{data.data()+Nsys+2, Nsys};
    }

    T* true_vector(){
        return data.data()+2;
    }

    T* exposed_vector(){
        return data.data() + Nsys+4;
    }

    void set_t(T t){
        data[0] = data[Nsys+2] = t;
    }

    void set_stepsize(T habs){
        data[1] = data[Nsys+3] = habs;
    }

    void set_true_vector(const T* vec){
        copy_array(data.data()+2, vec, Nsys);
    }

    void set_exposed_vector(const T* vec){
        copy_array(data.data()+Nsys+4, vec, Nsys);
    }

    inline void resize(size_t nsys){
        data.resize(nsys*2+4);
        Nsys = nsys;
    }

    inline size_t nsys() const{
        return Nsys;
    }
    
    bool choose_true = true; //if true, then exposed_vector may contain garbage values. Do not read its values. if false, then true_vector contains the true state vector, and exposed_vector contains the exposed state vector.
    bool triggered = false;

private:

    static constexpr size_t NTOT = (N > 0 ? 2*N+4 : 0);

    Array1D<T, NTOT, Allocation::Auto> data;
    size_t Nsys = 0;


};

// USEFUL ALIASES

template<typename T>
using Func = void(*)(T*, const T&, const T*, const T*, const void*); // f(t, q, args, void) -> array

template<typename T>
using FuncLike = void(*)(T*, const T&, const void*); // f(t, void) -> array

template<typename T>
using ObjFun = T(*)(const T&, const T*, const T*, const void*); // f(t, q, args, void) -> scalar

template<typename T>
using ObjFunLike = T(*)(const T&, const void*); // f(t, void) -> scalar

template<typename T, size_t N>
using JacMat = Array2D<T, N, N, Allocation::Heap, Layout::F>;

using EventMap = std::map<std::string, std::vector<size_t>>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag, std::complex;

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline TimePoint now(){
    return std::chrono::high_resolution_clock::now();
}

inline double as_duration(const TimePoint& t1, const TimePoint& t2){
    std::chrono::duration<double> duration = t2-t1;
    return duration.count();
}

template <typename T>
T inf() {
    // When using -ffast-math, infinity() may cause issues or segfaults
    // Use a very large finite number instead that's safe with -ffast-math
    #ifdef __FAST_MATH__
    return std::numeric_limits<T>::max();
    #else
    return std::numeric_limits<T>::infinity();
    #endif
}

template<typename T>
T norm_squared(const T* x, size_t size){
    //optimize
    T res = 0;
    #pragma omp simd reduction(+:res)
    for (size_t i=0; i<size; i++){
        res += x[i]*x[i];
    }
    return res;
}

template<typename T>
bool resize_step(T& factor, T& habs, const T& min_step, const T& max_step){
    bool res = false;
    if (habs*factor < min_step){
        factor = min_step/habs;
        habs = min_step;
    }
    else if (habs*factor > max_step){
        factor = max_step/habs;
        habs = max_step;
    }
    else{
        habs *= factor;
        res = true;
    }
    return res;
}

template <typename T>
inline bool is_finite(const T& value) {
    if constexpr (std::is_floating_point_v<T>) {
        #ifdef __FAST_MATH__
        // When -ffast-math is enabled, std::isfinite may not work correctly
        // Use range check instead: value is finite if it's within representable range
        return (value < std::numeric_limits<T>::max());
        #else
        return std::isfinite(value);
        #endif
    } else if constexpr (std::is_integral_v<T>) {
        return true;
    } else {
        static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
        return false;
    }
}

#ifdef MPREAL
template <>
inline bool is_finite(const mpfr::mpreal& value) {
    return mpfr_number_p(value.mpfr_ptr()) != 0;
}
#endif

template<typename T>
T rms_norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size)/size);
}

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size){
    T norm_sq = 0;
    #pragma omp simd reduction(+:norm_sq)
    for (size_t i=0; i<size; i++){
        norm_sq += x[i]*x[i]/(scale[i]*scale[i]);
    }
    return sqrt(norm_sq/size);
}

template<typename T>
T norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size));
}

template<typename T>
int sgn(const T& x){
    return ( x > 0) ? 1 : ( (x < 0) ? -1 : 0);
}

template<typename T>
int sgn(const T& t1, const T& t2){
    //same as sgn(t2-t1), but avoids roundoff error
    return (t1 < t2 ? 1 : (t1 > t2 ? -1 : 0));
}

template<typename T>
std::vector<T> subvec(const std::vector<T>& x, size_t start, size_t size) {
    if (start >= x.size()) {
        return {}; // Return an empty vector if start is out of bounds
    }
    return std::vector<T>(x.begin() + start, x.begin() + start + size);
}

template<typename T>
bool all_are_finite(const T* data, size_t n){
    for (size_t i=0; i<n; i++){
        if (!is_finite(data[i])){
            return false;
        }
    }
    return true;
}


template<typename T>
std::vector<T> _t_event_data(const T* t, const EventMap& event_map, const std::string& event){
    const std::vector<size_t>& ind = event_map.at(event);
    std::vector<T> data(ind.size());
    for (size_t i=0; i<data.size(); i++){
        data[i] = t[ind[i]];
    }
    return data;
}

template<typename T, size_t N>
Array2D<T, 0, N> _q_event_data(const T* q, const EventMap& event_map, const std::string& event, size_t Nsys){
    const std::vector<size_t>& ind = event_map.at(event);
    Array2D<T, 0, N> data(ind.size(), Nsys);
    for (size_t i=0; i<ind.size(); i++){
        for (size_t j=0; j<Nsys; j++){
            data(i, j) = q[ind[i]*Nsys+j];
        }
    }
    return data;
}

//BISECTION USED FOR EVENTS IN ODES

template<typename T>
std::vector<T> bisect(ObjFunLike<T> f, const T& a, const T& b, const T& atol, const void* obj){
    T err = 2*atol+1;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    if (f(a, obj)*f(b, obj) > 0){
        throw std::runtime_error("Root not bracketed");
    }

    while (err > atol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            break;
        }
        fm = f(c, obj);
        if (f(_a, obj) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = abs(fm);
    }
    return {_a, c, _b};
}


template<typename T>
void mat_vec_prod(T* result, const T* mat, const T* vec, size_t rows, size_t cols, const T& factor=1){
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

template<typename T>
void mat_T_vec_prod(T* result, const T* mat, const T* vec, size_t rows, size_t cols, const T& factor=1){
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
void mat_mat_prod(S* r, const S* a, const S* b, size_t m, size_t s, size_t n, const S& factor=1){
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
void mat_T_mat_prod(S* r, const S* a, const S* b, size_t m, size_t s, size_t n, const S& factor=1){
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


inline std::string format_duration(double t){
    int h = int(t/3600);
    int m = int((t - h*3600)/60);
    int s = int(t - h*3600 - m*60);

    return std::to_string(h) + " h, " + std::to_string(m) + " m, " + std::to_string(s) + " s";  
}

class Clock{

public:

    Clock()=default;

    inline void start(){
        _start = now();
    }

    inline double seconds() const{
        return as_duration(_start, now());
    }

    inline std::string message() const{
        return format_duration(seconds());
    }

private:

    TimePoint _start;
};


inline void show_progress(const int& n, const int& target, const Clock& clock){
    std::cout << "\033[2K\rProgress: " << std::setprecision(2) << n*100./target << "%" <<   " : " << n << "/" << target << "  Time elapsed : " << clock.message() << "      Estimated duration: " << format_duration(target*clock.seconds()/n) << std::flush;
}


template<typename T>
struct OdeData{

    Func<T> rhs=nullptr;
    Func<T> jacobian=nullptr;
    const void* obj = nullptr; //It will be passed inside rhs and jacobian

    /*
    IMPORTANT
    ================

    The jacobian function takes as first input the output matrix (just like the rhs function takes as input the output array).
    However, the output matrix stores its data in a column-major order, but is passed as the flattened array.
    Since the output matrix, must behave as J(i, j) = df_i / dx_j, the output array must be accessed
    as m[j + i*n] = df_i/dx_j, and not m[i + j*n].

    In other words, if the jacobian matrix is

    J_ij = [[a  b],
            [c  d]]
    
    Then the output array must be set as

    m[0] = a, m[1] = c, m[2] = b, m[3] = d
    and NOT
    m[0] = a, m[1] = b, m[2] = c, m[3] = d;
    */
};


//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<typename T, size_t N=0>
class OdeResult{

public:

    OdeResult(const std::vector<T>& t, const Array2D<T, 0, N>& q, EventMap event_map, bool diverges, bool success, double runtime, std::string message) : _t(t), _q(q), _event_map(std::move(event_map)), _diverges(diverges), _success(success), _runtime(runtime), _message(std::move(message)) {}

    DEFAULT_RULE_OF_FOUR(OdeResult);

    virtual ~OdeResult() = default;

    const std::vector<T>& t() const {return _t;}

    const Array2D<T, 0, N>& q() const {return _q;}

    template<std::integral INT1, std::integral INT2>
    const T& q(INT1 i, INT2 j) const {return _q(i, j);}

    const EventMap& event_map() const { return _event_map;}

    bool diverges() const {return _diverges;}

    bool success() const {return _success;}

    double runtime() const {return _runtime;}

    std::string message() const {return _message;}

    void examine() const{
        std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n" <<
        "\tPoints           : " << _t.size() << "\n" <<
        "\tDiverges         : " << (_diverges ? "true" : "false") << "\n" << 
        "\tSuccess          : " << (_success ? "true" : "false") << "\n" <<
        "\tRuntime          : " << _runtime << "\n" <<
        "\tTermination cause: " << _message << "\n" <<
        event_log();
    }

    std::string event_log() const{
        std::string res;
        res += "\tEvents:\n\t----------\n";
        for (const auto& [name, array] : _event_map){
            res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
        }
        res += "\n\t----------\n";
        return res;
    }

    std::vector<T> t_filtered(const std::string& event) const {
        return _t_event_data(this->_t.data(), this->_event_map, event);
    }

    Array2D<T, 0, N> q_filtered(const std::string& event) const {
        return _q_event_data<T, N>(this->_q.data(), this->_event_map, event, _q.Ncols());
    }

    virtual OdeResult<T, N>* clone() const{ return new OdeResult<T, N>(*this);}
    
private:

    std::vector<T> _t;
    Array2D<T, 0, N> _q;
    EventMap _event_map;
    bool _diverges;
    bool _success;// true if the OdeSolver didnt die during the integration
    double _runtime;
    std::string _message;
};

template<typename T>
inline T choose_step(const T& habs, const T& hmin, const T& hmax){
    return std::max(std::min(habs, hmax), hmin);
}


#endif