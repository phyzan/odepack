#pragma once

#include <utility>
#include <array>
#include <cstring>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <iostream>

#define THIS static_cast<Derived*>(this)
#define THIS_C static_cast<const Derived*>(this)
#define CONST_CAST(TYPE, FUNC) const_cast<TYPE>(static_cast<const CLS*>(this)->FUNC);
#define STATIC_IDX_ASSERT(idx, ND) static_assert(sizeof...(idx) == ND, "Incorrect number of indices")
#define DYNAMIC_IDX_ASSERT(idx, nd) assert(sizeof...(idx) == nd && "Incorrect number of indices")

template<typename... Args>
std::string idx_string(Args... arg){
    std::string s = "{ " + ((std::to_string(arg) + " ") + ...) + "}";
    return s;
}


template<typename... Ts>
concept INT_T = (std::is_integral_v<Ts>  && ...);

template<typename T>
inline void copy_array(T* dest, const T* src, size_t size){
    if (size==0) {return;}
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<typename T, size_t size>
inline void copy_array(T* dest, const T* src){
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<typename T>
inline bool equal_arrays(const T* a, const T* b, size_t size){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}

template<typename T>
std::string to_string(const T& value, int digits = 3) {
    static_assert(std::is_arithmetic<T>::value || std::is_class<T>::value, "T must be a numeric or class type with ostream << defined");

    std::ostringstream out;
    out << std::setprecision(digits) << std::scientific << value;
    return out.str();
}

template<typename T>
std::string array_repr(T* array, size_t size, int digits) {
    std::string result;
    if (size == 0){
        return result;
    }
    for (size_t i=0; i<size-1; i++){
        result += to_string(array[i], digits) + " ";
    }
    return result + to_string(array[size-1], digits);
}

template<typename T, size_t size>
inline bool equal_arrays(const T* a, const T* b){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}


template<size_t... Args>
size_t _validate_size(size_t size){
    constexpr size_t expected = (Args * ...);
    assert(size == expected && "Invalid initializer list size");
    return size;
}