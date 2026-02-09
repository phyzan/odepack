#ifndef TOOLS_IMPL_HPP
#define TOOLS_IMPL_HPP

#include "Tools.hpp"

namespace ode{

template<typename Type>
PolyWrapper<Type>::PolyWrapper(Type* object) : _ptr(object) {}

template<typename Type>
PolyWrapper<Type>::PolyWrapper(const PolyWrapper& other)
    : _ptr(other.new_ptr()) {}

template<typename Type>
PolyWrapper<Type>::PolyWrapper(PolyWrapper&& other) noexcept
    : _ptr(other.release()) {}

template<typename Type>
PolyWrapper<Type>& PolyWrapper<Type>::operator=(const PolyWrapper& other){
    if (&other != this){
        delete _ptr;
        _ptr = other.new_ptr();
    }
    return *this;
}

template<typename Type>
PolyWrapper<Type>& PolyWrapper<Type>::operator=(PolyWrapper&& other) noexcept{
    if (&other != this){
        delete _ptr;
        _ptr = other.release();
    }
    return *this;
}

template<typename Type>
PolyWrapper<Type>::~PolyWrapper(){
    delete _ptr;
    _ptr = nullptr;
}

template<typename Type>
Type* PolyWrapper<Type>::operator->(){
    assert(_ptr != nullptr && "pointer is null");
    return _ptr;
}

template<typename Type>
const Type* PolyWrapper<Type>::operator->() const{
    assert(_ptr != nullptr && "pointer is null");
    return _ptr;
}

template<typename Type>
const Type* PolyWrapper<Type>::ptr() const{
    return _ptr;
}

template<typename Type>
Type* PolyWrapper<Type>::ptr(){
    return _ptr;
}

template<typename Type>
Type* PolyWrapper<Type>::new_ptr() const{
    return _ptr == nullptr ? nullptr : _ptr->clone();
}

template<typename Type>
Type* PolyWrapper<Type>::release(){
    Type* tmp = _ptr;
    _ptr = nullptr;
    return tmp;
}

template<typename Type>
template<typename Base>
Base* PolyWrapper<Type>::cast(){
    return dynamic_cast<Base*>(this->_ptr);
}

template<typename Type>
void PolyWrapper<Type>::take_ownership(Type* ptr){
    delete _ptr;
    _ptr = ptr;
}


//===========================================================================================
//                                      State<T>
//===========================================================================================

template<typename T>
State<T>::State(const T* data, size_t Nsys)
    : _data(data), _nsys(Nsys) {}

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
State<T> EventState<T>::True() const{
    return {data.data(), Nsys};
}

template<typename T>
State<T> EventState<T>::exposed() const{
    return choose_true ? this->True()
                       : State<T>{data.data() + Nsys + 2, Nsys};
}

template<typename T>
T* EventState<T>::true_vector(){
    return data.data() + 2;
}

template<typename T>
T* EventState<T>::exposed_vector(){
    return data.data() + Nsys + 4;
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
//                                      OdeResult<T, N>
//===========================================================================================

template<typename T, size_t N>
OdeResult<T, N>::OdeResult(const std::vector<T>& t, const Array2D<T, 0, N>& q, EventMap event_map, bool diverges, bool success, double runtime, std::string message) : _t(t), _q(q), _event_map(std::move(event_map)), _diverges(diverges), _success(success), _runtime(runtime), _message(std::move(message)) {}



template<typename T, size_t N>
OdeResult<T, N>* OdeResult<T, N>::clone() const {
    return new OdeResult<T, N>(*this);
}

template<typename T, size_t N>
const std::vector<T>& OdeResult<T, N>::t() const {
    return _t;
}

template<typename T, size_t N>
const Array2D<T, 0, N>& OdeResult<T, N>::q() const {
    return _q;
}

template<typename T, size_t N>
const EventMap& OdeResult<T, N>::event_map() const {
    return _event_map;
}

template<typename T, size_t N>
bool OdeResult<T, N>::diverges() const {
    return _diverges;
}

template<typename T, size_t N>
bool OdeResult<T, N>::success() const {
    return _success;
}

template<typename T, size_t N>
double OdeResult<T, N>::runtime() const {
    return _runtime;
}

template<typename T, size_t N>
const std::string& OdeResult<T, N>::message() const {
    return _message;
}

template<typename T, size_t N>
void OdeResult<T, N>::examine() const {
    std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n"
              << "\tPoints           : " << _t.size() << "\n"
              << "\tDiverges         : " << (_diverges ? "true" : "false") << "\n"
              << "\tSuccess          : " << (_success ? "true" : "false") << "\n"
              << "\tRuntime          : " << _runtime << "\n"
              << "\tTermination cause: " << _message << "\n"
              << event_log();
}

template<typename T, size_t N>
std::string OdeResult<T, N>::event_log() const {
    std::string res;
    res += "\tEvents:\n\t----------\n";
    for (const auto& [name, array] : _event_map) {
        res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
    }
    res += "\n\t----------\n";
    return res;
}

template<typename T, size_t N>
std::vector<T> OdeResult<T, N>::t_filtered(const std::string& event) const {
    return _t_event_data(this->_t.data(), this->_event_map, event);
}

template<typename T, size_t N>
Array2D<T, 0, N> OdeResult<T, N>::q_filtered(const std::string& event) const {
    return _q_event_data<T, N>(this->_q.data(), this->_event_map, event, _q.Ncols());
}

template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax) {
    return std::max(std::min(habs, hmax), hmin);
}

//===========================================================================================
//                                      Additional Tools
//===========================================================================================

template<typename... Arg>
void print(Arg... x){
    ((std::cout << x << ' '), ...);
    std::cout << "\n";
}

template<typename A, typename B>
inline auto max(A a, B b) {
    return (a > b) ? a : b;
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

template <typename T>
bool is_finite(const T& value) {
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
bool is_finite(const mpfr::mpreal& value) {
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
        max_val = std::max(max_val, std::abs(x[i]));
    }
    return max_val;
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
        T max_val = ndspan::abs(lu[i*N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = ndspan::abs(lu[j*N + i]);
            if (val > max_val) { max_val = val; max_row = j; }
        }
        assert(max_val != 0 && "Matrix is singular");

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

} // namespace ode

#endif // TOOLS_IMPL_HPP