#ifndef VEC_HPP
#define VEC

#include <type_traits>
#include <cstddef>
#include <omp.h>  // Optional, only needed for `#pragma omp simd`



template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
apply_elementwise(T* res, const T* a, const T* b, size_t size, T (*func)(const T&, const T&)) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        res[i] = func(a[i], b[i]);
}


template<typename T>
typename std::enable_if<!std::is_arithmetic<T>::value>::type
apply_elementwise(T* res, const T* a, const T* b, size_t size, T (*func)(const T&, const T&)) {
    for (size_t i = 0; i < size; ++i)
        res[i] = func(a[i], b[i]);
}


template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
assign_elementwise(T* res, const T* other, size_t size) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        res[i] = other[i];
}


template<typename T>
typename std::enable_if<!std::is_arithmetic<T>::value>::type
assign_elementwise(T* res, const T* other, size_t size) {
    for (size_t i = 0; i < size; ++i)
        res[i] = other[i];
}


#endif