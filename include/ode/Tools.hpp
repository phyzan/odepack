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


namespace ode {

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag, std::complex;

using ndspan::Array, ndspan::Array1D, ndspan::Array2D, ndspan::View, ndspan::MutView, ndspan::View1D, ndspan::Allocation, ndspan::Layout, ndspan::prod, ndspan::copy_array, ndspan::to_string, ndspan::abs;

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

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

using VoidType = void(*)();

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

enum class RootPolicy : std::uint8_t { Left, Middle, Right};

template<typename Type>
class PolyWrapper{

    /*
    Takes ownership of a pointer to a heap-allocated object.
    Is constructed by passing the pointer, so make sure
    noone else has ownership.

    MUST:
        Type has a clone() method that returns a new Type*,
        so that it creates a perfect copy.
    */

public:
    
    PolyWrapper(Type* object);

    PolyWrapper(const PolyWrapper& other);

    PolyWrapper(PolyWrapper&& other) noexcept;

    PolyWrapper& operator=(const PolyWrapper& other);

    PolyWrapper& operator=(PolyWrapper&& other) noexcept;

    ~PolyWrapper();
    
    Type* operator->();

    const Type* operator->() const;

    const Type* ptr() const;

    Type* ptr();

    Type* new_ptr() const;

    Type* release();

    template<typename Base>
    Base* cast();

    void take_ownership(Type* ptr);

    PolyWrapper() = default;
    
protected:

    Type* _ptr = nullptr;
};



template<typename T>
class State{

    //provides a view of data, does not own it. The lifetime of a State object must be shorter
    //than that of the underlying data, otherwise the program will crash or behave incorrectly

public:

    State(const T* data, size_t Nsys);

     const T& t() const;

     const T& habs() const;

     const T* vector() const;

     size_t Nsys() const;

protected:

    const T* _data;
    size_t _nsys;
};

template<typename T>
class MutState : State<T>{

    MutState(T* data, size_t Nsys);

    T* vector();

};


template<typename T>
class EventState{

    Array1D<T> data;
    size_t Nsys = 0;

public:

    EventState() = default;

    const T& t() const;

    State<T> True() const;

    State<T> exposed() const;

    T* true_vector();

    T* exposed_vector();

    void set_t(T t);

    void set_stepsize(T habs);

    void set_true_vector(const T* vec);

    void set_exposed_vector(const T* vec);

    void resize(size_t nsys);

    size_t nsys() const;

    bool is_valid() const;
    
    bool choose_true = true; //if true, then exposed_vector may contain garbage values. Do not read its values. if false, then true_vector contains the true state vector, and exposed_vector contains the exposed state vector.
    bool triggered = false;
};

//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<typename T, size_t N=0>
class OdeResult{

public:

    OdeResult(const std::vector<T>& t, const Array2D<T, 0, N>& q, EventMap event_map, bool diverges, bool success, double runtime, std::string message);

    OdeResult() = default;

    DEFAULT_RULE_OF_FOUR(OdeResult);

    virtual ~OdeResult() = default;

    const std::vector<T>& t() const;

    const Array2D<T, 0, N>& q() const;

    template<std::integral INT1, std::integral INT2>
    const T& q(INT1 i, INT2 j) const;

    const EventMap& event_map() const;

    bool diverges() const;

    bool success() const;

    double runtime() const;

    const std::string& message() const;

    void examine() const;

    std::string event_log() const;

    std::vector<T> t_filtered(const std::string& event) const;

    Array2D<T, 0, N> q_filtered(const std::string& event) const;

    virtual OdeResult<T, N>* clone() const;
    
private:

    std::vector<T> _t;
    Array2D<T, 0, N> _q;
    EventMap _event_map;
    bool _diverges = false;
    bool _success = false;
    double _runtime = 0;
    std::string _message = "No integration performed";
};

template<typename RHS, typename JAC = std::nullptr_t>
struct OdeData {
    RHS rhs;
    JAC jacobian = nullptr;
    const void* obj = nullptr;

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



template<typename RHS>
struct OdeData<RHS, void> {

    RHS rhs;
    VoidType jacobian = [](){}; //will be cast to Func<T> and will be checked for nullptr at runtime
    const void* obj = nullptr;
};


class Clock{

public:

    Clock() = default;

    inline static TimePoint now(){
        return std::chrono::high_resolution_clock::now();
    }

    inline static double as_duration(const TimePoint& t1, const TimePoint& t2){
        std::chrono::duration<double> duration = t2-t1;
        return duration.count();
    }


    inline static std::string format_duration(double t){
        int h = int(t/3600);
        int m = int((t - h*3600)/60);
        int s = int(t - h*3600 - m*60);

        return std::to_string(h) + " h, " + std::to_string(m) + " m, " + std::to_string(s) + " s";  
    }

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

template<typename A, typename B>
auto max(A a, B b);

template<typename T>
T inf();

template<typename T>
T norm_squared(const T* x, size_t size);

template<typename T>
bool resize_step(T& factor, T& habs, const T& min_step, const T& max_step);

template<typename T>
bool is_finite(const T& value);

#ifdef MPREAL
template <>
bool is_finite(const mpfr::mpreal& value);
#endif

template<typename T>
T rms_norm(const T* x, size_t size);

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size);

template<typename T>
T inf_norm(const T* x, size_t size);

template<typename T>
T norm(const T* x, size_t size);

template<typename T>
int sgn(const T& x);

template<typename T>
int sgn(const T& t1, const T& t2);

template<typename T>
std::vector<T> subvec(const std::vector<T>& x, size_t start, size_t size);

template<typename T>
bool all_are_finite(const T* data, size_t n);

template<typename T>
std::vector<T> _t_event_data(const T* t, const EventMap& event_map, const std::string& event);

template<typename T, size_t N>
Array2D<T, 0, N> _q_event_data(const T* q, const EventMap& event_map, const std::string& event, size_t Nsys);

template<typename T, RootPolicy RP, typename Callable>
T bisect(Callable&& f, const T& a, const T& b, const T& atol);

template<typename T>
void inv_mat_row_major(T* out, const T* mat, size_t N, T* work, size_t* pivot);


inline void show_progress(int n, int target, const Clock& clock){
    std::cout << "\033[2K\rProgress: " << std::setprecision(2) << n*100./target << "%" <<   " : " << n << "/" << target << "  Time elapsed : " << clock.message() << "      Estimated duration: " << Clock::format_duration(target*clock.seconds()/n) << std::flush;
}




template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax);

} // namespace ode

#endif