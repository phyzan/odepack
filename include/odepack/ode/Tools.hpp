#ifndef TOOLS_HPP
#define TOOLS_HPP


#include <complex>
#include <map>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <autodiff/autodiff.hpp>
#include "../polybox/polybox.hpp"

#ifdef MPREAL
#include <mpreal.h>
#endif

#include "../ndspan/arrays.hpp"


namespace ode {

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag, ndspan::min, ndspan::max, std::complex;

using ndspan::Array, ndspan::Array1D, ndspan::Array2D, ndspan::View, ndspan::MutView, ndspan::View1D, ndspan::View2D, ndspan::View3D, ndspan::Allocation, ndspan::Layout, ndspan::prod, ndspan::copy_array, ndspan::copy_array, ndspan::to_string, ndspan::abs;

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

// USEFUL ALIASES

template<typename T>
using RhsFunc = void(*)(T*, const T&, const T*, const T*); // f(t, q, args) -> array

template<typename T>
using ObjFun = T(*)(const T&, const T*, const T*); // f(t, q, args) -> scalar

template<typename F, typename T>
concept isRhsFunc = 
requires(F f, T* out, T t, const T* q, const T* args){
    { f(out, t, q, args) } -> std::same_as<void>;
    { f(out, std::as_const(t), q, args) } -> std::same_as<void>;
};

template<typename F, typename T>
concept OptionalRhsFunc = std::same_as<F, std::nullptr_t> || isRhsFunc<F, T>;

template<typename F, typename T>
concept isObjFun =
requires(F f, T t, const T* q, const T* args){
    { f(t, q, args) } -> std::convertible_to<T>;
    { f(std::as_const(t), q, args) } -> std::convertible_to<T>;
};


template<typename F, typename T>
concept Observer =
requires(F f, T t, const T* q, const T* t_ptr){
    { f(t, q, t_ptr) } -> std::convertible_to<bool>;
    { f(std::as_const(t), q, t_ptr) } -> std::convertible_to<bool>;
};

template<typename F, typename T>
concept OptionalObserver = std::is_same_v<F, std::nullptr_t> || Observer<F, T>;

template<typename F, typename T>
concept StateInterp = requires(F f, T* out, T t){
    { f(out, t) } -> std::same_as<void>;
    { f(out, std::as_const(t)) } -> std::same_as<void>;
};




template<typename T, size_t N>
using JacMat = Array2D<T, N, N, Allocation::Auto, Layout::F>;

using VoidType = void(*)();

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

enum class RootPolicy : std::uint8_t { Left, Middle, Right};

template<typename T>
bool allEqual(const T* a, const T* b, size_t n);

template<typename T, RootPolicy RP, typename Callable>
T bisect(Callable&& f, const T& a, const T& b, const T& atol);

template<typename T>
void inv_mat_row_major(T* out, const T* mat, size_t N, T* work, size_t* pivot);

template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax);

template<typename T>
T detLU_row_major(T* mat, size_t N);

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


template<typename RHS, typename JAC = std::nullptr_t>
struct OdeData {

    using RhsType = RHS;
    using JacType = JAC;

    RHS Rhs;
    JAC Jac = nullptr;

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


// Check if F has a callable Rhs (static or non-static, non-template)
template<typename F, typename T>
concept hasRhsFunc =
    requires(F f, T* out, T t, const T* q, const T* args) {
        { f.Rhs(out, t, q, args) } -> std::same_as<void>;
        { f.Rhs(out, std::as_const(t), q, args) } -> std::same_as<void>;
    };

// Check if F has a callable Jac (static or non-static, non-template)
template<typename F, typename T>
concept hasJacFunc =
    requires(F f, T* out, T t, const T* q, const T* args) {
        { f.Jac(out, t, q, args) } -> std::same_as<void>;
        { f.Jac(out, std::as_const(t), q, args) } -> std::same_as<void>;
    };

// Check if F has a callable templated Rhs (static or non-static)
template<typename F, typename T>
concept hasTemplateRhs =
    requires(F f, T* out, T t, const T* q, const T* args) {
        { f.template Rhs<T>(out, t, q, args) } -> std::same_as<void>;
        { f.template Rhs<T>(out, std::as_const(t), q, args) } -> std::same_as<void>;
    };

// Check if F has a callable templated Jac (static or non-static)
template<typename F, typename T>
concept hasTemplateJac =
    requires(F f, T* out, T t, const T* q, const T* args) {
        { f.template Jac<T>(out, t, q, args) } -> std::same_as<void>;
        { f.template Jac<T>(out, std::as_const(t), q, args) } -> std::same_as<void>;
    };

template<typename F, typename T>
concept hasRhsOnly = hasRhsFunc<F, T> && !hasJacFunc<F, T>;

template<typename F, typename T>
concept hasRhsAndJac = hasRhsFunc<F, T> && hasJacFunc<F, T>;


enum class JacPolicy : std::uint8_t{
    Approx,
    Exact,
    Autodiff,
};


template<typename T, hasRhsFunc<T> F>
constexpr JacPolicy getJacPolicy(){
    if constexpr (hasTemplateRhs<F, T> && !hasJacFunc<F, T>){
        return JacPolicy::Autodiff;
    } else if (hasJacFunc<F, T>){
        return JacPolicy::Exact;
    } else {
        return JacPolicy::Approx;
    }
}


enum class StepResult : std::uint8_t {
    Success, // Successful step
    INF_ERROR, // Non-finite value encountered (e.g., NaN or Inf)
    TINY_STEP_ERROR, // Step size became too small (below machine epsilon)
    MIN_STEP_ERROR, // Step size reached minimum set by user
    MAX_STEP_ERROR, // Step size reached maximum set by user
};

template<typename T>
struct EmptyArr{

    EmptyArr() = default;

    inline T operator[](size_t) const{
        return T(0);
    }

    inline size_t size() const{
        return 0;
    }

    inline const T* data() const{
        return nullptr;
    }
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
inline bool isfinite(const T& value) {
#ifndef NO_NAN_CHECK
    if constexpr (!std::is_integral_v<T>) {
        #ifdef __FAST_MATH__
        // When -ffast-math is enabled, std::isfinite may not work correctly
        // Use range check instead: value is finite if it's within representable range
        return (value >= std::numeric_limits<T>::lowest() &&
                value <= std::numeric_limits<T>::max());
        #else
        return std::isfinite(value);
        #endif
    } else {
        return true; // Integral types are always finite
    }
#else
    return true; // If NO_NAN_CHECK is defined, assume all values are finite
#endif
}


template<typename T>
T rms_norm(const T* x, size_t size);

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size);

template<typename T>
T inf_norm(const T* x, size_t size);

template<typename T>
T norm(const T* x, size_t size);


template<typename T>
std::vector<T> subvec(const std::vector<T>& x, size_t start, size_t size);

template<typename T>
INLINE bool all_are_finite(const T* data, size_t n){
#ifndef NO_NAN_CHECK
    for (size_t i=0; i<n; i++){
        if (!isfinite(data[i])){
            return false;
        }
    }
#endif
    return true;
}



inline void show_progress(int n, int target, const Clock& clock){
    std::cout << "\033[2K\rProgress: " << std::setprecision(2) << n*100./target << "%" <<   " : " << n << "/" << target << "  Time elapsed : " << clock.message() << "      Estimated duration: " << Clock::format_duration(target*clock.seconds()/n) << std::flush;
}

template<typename... Arg>
inline void print(Arg... x){
    ((std::cout << x << ' '), ...);
    std::cout << "\n";
}


} // namespace ode

#endif