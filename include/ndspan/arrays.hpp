#pragma once

#include "ndview.hpp"

template<typename T, Layout L, size_t... DIMS>
class Array : public AbstractNdView<Array<T, L, DIMS...>, L, T, DIMS...>{

    using CLS = Array<T, L, DIMS...>;
    using Base = AbstractNdView<Array<T, L, DIMS...>, L, T, DIMS...>;

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

public:

    inline static constexpr bool IS_HEAP = ((N==0) || (sizeof(T)*N > 80000));

    Array() {
        if constexpr (IS_HEAP && N > 0) {
            _dynamic_arr = new T[N];
        }
    }

    explicit Array(const T* arr) requires (N>0) {
        if constexpr (IS_HEAP) {
            _dynamic_arr = new T[N];
        }
        copy_array<T, N>(data(), arr);
    }
    
    template<INT_T... Args>
    explicit Array(Args... shape) : Base(shape...) {
        static_assert(sizeof...(shape) > 0, "Cannot construct shape from no shape");
        if constexpr (IS_HEAP) {
            if (this->size() > 0){
                _dynamic_arr = new T[this->size()];
            }
        }
    }

    template<INT_T... Args>
    explicit Array(const T* arr, Args... shape) : Array(shape...){
        if (this->size() > 0){
            copy_array(this->data(), arr, this->size());
        }
    }

    Array(std::initializer_list<T> array) requires (ND<2 && (N == 0)) : Array(array.begin(), array.size()) {}

    Array(std::initializer_list<T> array) requires (N>0) : Array(array.begin(), (_validate_size<DIMS...>(array.size()), DIMS)...) {}

    //COPY CONSTRUCTORS
    Array(const Array& other) : Base(other), _dynamic_arr((IS_HEAP && other.size() > 0) ? new T[other.size()] : nullptr) {
        copy_array(data(), other.data(), this->size());
    }

    //MOVE CONSTRUCTORS
    Array(Array&& other) noexcept : Base(std::move(other)), _dynamic_arr(IS_HEAP ? other._dynamic_arr : nullptr) {
        if constexpr (IS_HEAP) { other._dynamic_arr = nullptr;}
        else{
            copy_array(data(), other.data(), this->size());
        }
    }

    //ASSIGNMENT OPERATORS
    Array& operator=(const Array& other) {
        if (&other != this){
            if constexpr (IS_HEAP){
                if (this->size() != other.size()){
                    delete[] _dynamic_arr;
                    _dynamic_arr = other.size() > 0 ? new T[other.size()] : nullptr;                
                }
            }
            Base::operator=(other);
            copy_array(data(), other.data(), this->size());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS

    Array& operator=(Array&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            if constexpr (IS_HEAP){
                delete[] _dynamic_arr;
                _dynamic_arr = other._dynamic_arr;
                other._dynamic_arr = nullptr;
            }
            else{
                copy_array(data(), other.data(), this->size());
            }
        }
        return *this;
    }

    Array& set(const T& value){
        size_t s = this->size();
        T* d = this->data();
        #pragma omp simd
        for (size_t i=0; i<s; i++){
            d[i] = value;
        }
        return *this;
    }

    ~Array() {
        if constexpr (IS_HEAP){
            delete[] _dynamic_arr;
            _dynamic_arr = nullptr;
        }
    }

    inline const T* data() const{
        if constexpr (IS_HEAP){
            return _dynamic_arr;
        }
        else{
            return _fixed_arr;
        }
    }

    inline T* data() {
        if constexpr (IS_HEAP){
            return _dynamic_arr;
        }
        else{
            return _fixed_arr;
        }
    }

    template<INT_T... Size>
    void resize(Size... newsize){
        size_t current_size = this->size();
        Base::resize(newsize...);
        if constexpr (IS_HEAP){
            size_t total_size = (newsize * ...);
            if (total_size != current_size){
                delete[] _dynamic_arr;
                if (total_size == 0){
                    _dynamic_arr = nullptr;
                }
                else{
                    _dynamic_arr = new T[total_size];
                }
                
            }
        }
        
    }

    inline std::string repr(int digits=8) const {
        return array_repr(this->data(), this->size(), digits);
    }

protected:

    T* _dynamic_arr = nullptr;
    T _fixed_arr[IS_HEAP ? 1 : N];
};


template<typename T, size_t Nr, size_t Nc, Layout L = Layout::C>
class Array2D : public Array<T, L, Nr, Nc>{

public:
    using Array<T, L, Nr, Nc>::Array;

    inline size_t Nrows() const {return this->shape()[0];}

    inline size_t Ncols() const {return this->shape()[1];}

    std::string repr(int digits=8) const {
        if (this->size() == 0){
            return "[]";
        }
        std::vector<size_t> column_str_len(this->Ncols());
        for (size_t j=0; j<this->Ncols(); j++){
            size_t len = to_string((*this)(0, j), digits).size();
            for (size_t i=0; i<this->Nrows(); i++){
                len = std::max(len, to_string((*this)(i, j), digits).size());
            }
            column_str_len[j] = len;
        }
        
        std::string result, element;
        for (size_t i=0; i<this->Nrows(); i++){
            for (size_t j=0; j<this->Ncols(); j++){
                element = to_string((*this)(i, j), digits);
                result += std::string(column_str_len[j]-element.size(), ' ') + element;
                if (j < this->Ncols()-1){
                    result += " ";
                }
            }
            if (i<this->Nrows()-1){
                result += "\n";
            }
        }
        return result;
    }
};


//alias
template<typename T, size_t N=0, Layout L = Layout::C>
using Array1D = Array<T, L, N>;