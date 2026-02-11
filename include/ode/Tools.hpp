#ifndef TOOLS_HPP
#define TOOLS_HPP


#include <complex>
#include <map>
#include <chrono>
#include <omp.h>
#include <cmath>
#include "../ndspan/arrays.hpp"
#ifdef MPREAL
#include <mpreal.h>
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

    inline ~PolyWrapper(){ delete _ptr;}
    
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

    State<T> true_state() const;

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

template<typename T>
class StepSequence{

public:

    StepSequence() = default;

    StepSequence(T* data, long int size, bool own_it = false);

    template<std::integral INT>
    inline const T& operator[](INT i) const{
        return _data[i];
    }

    StepSequence(const StepSequence& other);

    StepSequence(StepSequence&& other) noexcept;

    ~StepSequence();

    StepSequence& operator=(const StepSequence& other) = delete;

    StepSequence& operator=(StepSequence&& other) noexcept = delete;

    long int size() const;

    const T* data() const;

private:

    T* _data = nullptr;
    long int _size = -1;
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
inline auto max(A a, B b) {
    return (a > b) ? a : b;
}

template <typename T>
inline T inf() {
    // When using -ffast-math, infinity() may cause issues or segfaults
    // Use a very large finite number instead that's safe with -ffast-math
    #ifdef __FAST_MATH__
    return std::numeric_limits<T>::max();
    #else
    return std::numeric_limits<T>::infinity();
    #endif
}

template<typename T>
T norm_squared(const T* x, size_t size);

template<typename T>
bool resize_step(T& factor, T& habs, const T& min_step, const T& max_step);

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
T rms_norm(const T* x, size_t size);

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size);

template<typename T>
T inf_norm(const T* x, size_t size);

template<typename T>
T norm(const T* x, size_t size);

template<typename T>
inline int sgn(const T& x){
    return ( x > 0) ? 1 : ( (x < 0) ? -1 : 0);
}

template<typename T>
inline int sgn(const T& t1, const T& t2){
    //same as sgn(t2-t1), but avoids roundoff error
    return (t1 < t2 ? 1 : (t1 > t2 ? -1 : 0));
}

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

template<typename... Arg>
inline void print(Arg... x){
    ((std::cout << x << ' '), ...);
    std::cout << "\n";
}

template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax);

} // namespace ode

#endif