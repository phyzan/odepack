#ifndef TOOLS_IMPL_HPP
#define TOOLS_IMPL_HPP

#include "Tools.hpp"

namespace ode{

//===========================================================================================
//                                      State<T>
//===========================================================================================

template<typename T>
State<T>::State(const T* data, size_t Nsys) : _data(data), _nsys(Nsys) {}

template<typename T>
const T& State<T>::t() const{
    return _data[0];
}

template<typename T>
const T& State<T>::habs() const{
    return _data[1];
}

template<typename T>
const T* State<T>::vector() const{
    return _data + 2;
}

template<typename T>
size_t State<T>::Nsys() const{
    return _nsys;
}


//===========================================================================================
//                                      MutState<T>
//===========================================================================================

template<typename T>
MutState<T>::MutState(T* data, size_t Nsys)
    : State<T>(data, Nsys) {}

template<typename T>
T* MutState<T>::vector(){
    return const_cast<T*>(this->_data) + 2;
}

//===========================================================================================
//                                      EventState<T>
//===========================================================================================

template<typename T>
const T& EventState<T>::t() const{
    return data[0];
}

template<typename T>
const T* EventState<T>::get_true() const{
    return data.data();
}

template<typename T>
const T* EventState<T>::get_exposed() const{
    return choose_true ? data.data() : data.data() + Nsys + 2;
}

template<typename T>
T* EventState<T>::mut_true(){
    return data.data();
}

template<typename T>
T* EventState<T>::mut_exposed(){
    return data.data() + Nsys + 2;
}

template<typename T>
void EventState<T>::set_t(T t){
    data[0] = data[Nsys + 2] = t;
}

template<typename T>
void EventState<T>::set_stepsize(T habs){
    data[1] = data[Nsys + 3] = habs;
}

template<typename T>
void EventState<T>::set_true_vector(const T* vec){
    copy_array(data.data() + 2, vec, Nsys);
}

template<typename T>
void EventState<T>::set_exposed_vector(const T* vec){
    copy_array(data.data() + Nsys + 4, vec, Nsys);
}

template<typename T>
void EventState<T>::resize(size_t nsys){
    data.resize(nsys * 2 + 4);
    Nsys = nsys;
}

template<typename T>
size_t EventState<T>::nsys() const{
    return Nsys;
}

template<typename T>
bool EventState<T>::is_valid() const{
    return triggered;
}


//===========================================================================================
//                                      Additional Tools
//===========================================================================================



template<typename T>
T norm_squared(const T* x, size_t size){
    //optimize
    T res = 0;
    // #pragma omp simd reduction(+:res)
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



template<typename T>
T rms_norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size)/size);
}

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size){
    T norm_sq = 0;
    // #pragma omp simd reduction(+:norm_sq)
    for (size_t i=0; i<size; i++){
        norm_sq += x[i]*x[i]/(scale[i]*scale[i]);
    }
    return sqrt(norm_sq/size);
}

template<typename T>
T inf_norm(const T* x, size_t size){
    T max_val = 0;
    for (size_t i = 0; i < size; i++){
        max_val = ndspan::max<T>(max_val, std::abs(x[i]));
    }
    return max_val;
}

template<typename T>
T norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size));
}

template<typename T>
std::vector<T> subvec(const std::vector<T>& x, size_t start, size_t size) {
    if (start >= x.size()) {
        return {}; // Return an empty vector if start is out of bounds
    }
    return std::vector<T>(x.begin() + start, x.begin() + start + size);
}


template<typename T>
bool allEqual(const T* a, const T* b, size_t n){
    for (size_t i=0; i<n; i++){
        if (a[i] != b[i]){
            return false;
        }
    }
    return true;
}


//BISECTION USED FOR EVENTS IN ODES

template<typename T, RootPolicy RP, typename Callable>
T bisect(Callable&& f, const T& a, const T& b, const T& atol){
    T err = 2*atol+1;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    assert((f(a) * f(b) <= 0) && "Root not bracketed" );
    
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

    if constexpr (RP == RootPolicy::Left) {
        return _a;
    }else if constexpr (RP == RootPolicy::Middle) {
        return c;
    }else {
        return _b;
    }
}

template<typename T>
void inv_mat_row_major(T* out, const T* mat, size_t N, T* work, size_t* pivot) {
    // out: size N*N, row-major order
    // mat: size N*N, row-major order
    // work: size N
    // pivot: size N
    if (N == 0){
        return;
    }

    std::vector<T> lu(N * N);
    for (size_t i = 0; i < N * N; ++i){
        lu[i] = mat[i];
    }

    for (size_t i = 0; i < N; ++i){
        pivot[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        size_t max_row = i;
        T max_val = abs(lu[i*N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs(lu[j*N + i]);
            if (val > max_val) { max_val = val; max_row = j; }
        }
        // assert(max_val != 0 && "Matrix is singular");

        if (max_row != i) {
            for (size_t k = 0; k < N; ++k){
                std::swap(lu[i*N + k], lu[max_row*N + k]);
            }
            std::swap(pivot[i], pivot[max_row]);
        }

        for (size_t j = i + 1; j < N; ++j) {
            lu[j*N + i] /= lu[i*N + i];
            for (size_t k = i + 1; k < N; ++k){
                lu[j*N + k] -= lu[j*N + i] * lu[i*N + k];
            }
        }
    }

    for (size_t col = 0; col < N; ++col) {
        for (size_t i = 0; i < N; ++i){
            work[i] = (pivot[i] == col) ? 1 : 0;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
        }

        for (size_t i = N; i-- > 0;) {
            for (size_t j = i + 1; j < N; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
            work[i] /= lu[i*N + i];
        }

        for (size_t i = 0; i < N; ++i){
            out[i*N + col] = work[i];
        }

    }
}


template<typename T>
T detLU_row_major(T* mat, size_t N) {
    if (N == 0) {
        return 1;
    }

    T det = 1;
    int sign = 1;

    for (size_t i = 0; i < N; ++i) {
        // Partial pivoting
        size_t pivot = i;
        T max_val = abs(mat[i * N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs(mat[j * N + i]);
            if (val > max_val) {
                pivot = j;
                max_val = val;
            }
        }

        if (max_val == T(0)) {
            return T(0);
        }

        if (pivot != i) {
            // Swap rows
            for (size_t k = 0; k < N; ++k) {
                T tmp = mat[i * N + k];
                mat[i * N + k] = mat[pivot * N + k];
                mat[pivot * N + k] = tmp;
            }
            sign = -sign;
        }

        det *= mat[i * N + i];

        // Eliminate below pivot
        for (size_t j = i + 1; j < N; ++j) {
            T factor = mat[j * N + i] / mat[i * N + i];
            for (size_t k = i; k < N; ++k) {
                mat[j * N + k] = mat[j * N + k] - factor * mat[i * N + k];
            }
        }
    }

    return det*sign;
}

template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax) {
    return ndspan::max<T>(ndspan::min<T>(habs, hmax), hmin);
}

} // namespace ode

#endif // TOOLS_IMPL_HPP