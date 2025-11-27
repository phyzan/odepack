#pragma once

#include <utility>
#include <array>
#include <cstring>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <iostream>
#include "mpreal.h"

#define THIS static_cast<Derived*>(this)
#define THIS_C static_cast<const Derived*>(this)
#define CONST_CAST(TYPE, FUNC) const_cast<TYPE>(static_cast<const CLS*>(this)->FUNC);
#define INLINE __attribute__((always_inline)) inline

#define DEFAULT_RULE_OF_FOUR(CLASSNAME)                  \
    CLASSNAME(const CLASSNAME& other) = default;      \
    CLASSNAME(CLASSNAME&& other) = default;           \
    CLASSNAME& operator=(const CLASSNAME& other) = default; \
    CLASSNAME& operator=(CLASSNAME&& other) = default;

template<typename _ShapeContainer>
concept IsShapeContainer = requires(_ShapeContainer c) {
    { c.data() } -> std::same_as<std::remove_cv_t<std::remove_reference_t<decltype(c.data())>>>;
} && std::is_pointer_v<std::decay_t<decltype(std::declval<_ShapeContainer>().data())>> &&
    std::is_integral_v<std::remove_pointer_t<std::decay_t<decltype(std::declval<_ShapeContainer>().data())>>>;

template<typename Container>
inline void constexpr assert_integral_data(const Container& obj){
    using DataType = decltype(obj.data());

    // Ensure it's a pointer
    static_assert(std::is_pointer_v<DataType>, "data() must return a pointer");

    // Ensure it's a pointer to const
    static_assert(std::is_const_v<std::remove_pointer_t<DataType>>,
                  "data() must return a pointer to const");

    // Ensure the pointee is an integral type
    static_assert(std::is_integral_v<std::remove_pointer_t<DataType>>,
                  "data() must return a pointer to an integral type");
}
    
template<typename... Args>
auto max_of_pack(Args... args) {
    return (std::max)({args...});
}

template<typename... Args>
std::string idx_string(Args... arg){
    std::string s = "{ " + ((std::to_string(arg) + " ") + ...) + "}";
    return s;
}

template<typename Iterable>
size_t prod(const Iterable& array){
    if (array.size() == 0){
        return 0;
    }
    size_t res = 1;
    for (size_t i=0; i<array.size(); i++){
        res *= array[i];
    }
    return res;
}


template<typename... Ts>
concept INT_T = (std::is_integral_v<Ts>  && ...);

template<typename T>
INLINE void copy_array(T* dest, const T* src, size_t size){
    if (size==0) {return;}
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<std::integral INT_DST, std::integral INT_SRC>
requires (!std::same_as<INT_DST, INT_SRC>)
INLINE void copy_array(INT_DST* dest, const INT_SRC* src, size_t size) {
    if (size == 0) {return;}
    for (size_t i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

template<typename T>
INLINE bool equal_arrays(const T* a, const T* b, size_t size){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}

template<typename T>
std::string to_string(const T& value, int prec = 3) {
    static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, mpfr::mpreal>, 
                  "T must be numeric or mpreal");

    if constexpr (std::is_integral_v<T>) {
        // Integral types
        return std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        // Floating-point types (double, float, long double)
        std::ostringstream out;
        out << std::setprecision(prec) << std::scientific << value;
        return out.str();
    }
}

template<>
std::string to_string(const mpfr::mpreal& value, int prec){
    return value.toString(prec);
}

template<typename T, typename Array>
std::string array_repr(const Array& array, int digits) {
    std::string result;
    if (array.size() == 0){
        return result;
    }
    for (size_t i=0; i<array.size()-1; i++){
        result += to_string(array[i], digits) + " ";
    }
    return result + to_string(array[array.size()-1], digits);
}

template<typename T, size_t size>
INLINE bool equal_arrays(const T* a, const T* b){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}


template<size_t... Args>
size_t _validate_size(size_t size){
    assert((size == (Args * ...)) && "Invalid initializer list size");
    return size;
}


template<size_t...>
struct tail_product {
    static constexpr size_t value = 1;
};

// Recursive case
template<size_t Head, size_t... Tail>
struct tail_product<Head, Tail...> {
    static constexpr size_t value = Head * tail_product<Tail...>::value;
};
