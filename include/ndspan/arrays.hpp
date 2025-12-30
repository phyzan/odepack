#pragma once

#include "ndview.hpp"


template<typename Derived, typename T, Layout L, size_t... DIMS>
class AbstractArray : public AbstractNdView<Derived, L, T, DIMS...>{

    using CLS = AbstractArray<Derived, T, L, DIMS...>;
    using Base = AbstractNdView<Derived, L, T, DIMS...>;

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

public:

    Derived& set(const T& value){
        std::fill(this->begin(), this->end(), value);
        return static_cast<Derived&>(*this);
    }

protected:

    using Base::Base;

    AbstractArray() = default;

    DEFAULT_RULE_OF_FOUR(AbstractArray)

    template<IsShapeContainer ShapeContainer>
    explicit AbstractArray(const ShapeContainer& shape) : Base(shape) {}

    ~AbstractArray() = default;

    void _copy_from(const T* data){
        if (this->size() > 0){
            copy_array(this->data(), data, this->size());
        }
    }

};


template<typename T, Layout L, size_t... DIMS>
class DynamicArray : public AbstractArray<DynamicArray<T, L, DIMS...>, T, L, DIMS...>{

    using CLS = DynamicArray<T, L, DIMS...>;
    using Base = AbstractArray<DynamicArray<T, L, DIMS...>, T, L, DIMS...>;

public:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

    DynamicArray() : Base() {
        _data = (N > 0 ? new T[N] : nullptr);
    }

    explicit DynamicArray(const T* data) requires (N>0) : DynamicArray(){
        this->_copy_from(data);
    }

    template<INT_T... Args>
    explicit DynamicArray(Args... shape) : Base(shape...) {
        if (this->size() > 0){
            _data = new T[this->size()];
        }
    }

    template<IsShapeContainer ShapeContainer>
    explicit DynamicArray(const ShapeContainer& shape) : Base(shape){
        if (this->size() > 0){
            _data = new T[this->size()];
        }
    }

    template<IsShapeContainer ShapeContainer>
    explicit DynamicArray(const T* data, const ShapeContainer& shape) : DynamicArray(shape){
        copy_array<T>(this->data(), data, this->size());
    }

    template<IsShapeContainer ShapeContainer>
    explicit DynamicArray(T* data, const ShapeContainer& shape, bool own_it = false) : Base(shape){
        if (own_it){
            _data = data;
        }
        else if (this->size() > 0){
            _data = new T[this->size()];
            copy_array<T>(this->data(), data, this->size());
        }
    }

    template<INT_T... Args>
    explicit DynamicArray(const T* data, Args... shape) : DynamicArray(shape...){
        this->_copy_from(data);
    }

    DynamicArray(std::initializer_list<T> array) requires (ND<2 && (N == 0)) : DynamicArray(array.begin(), array.size()) {}

    DynamicArray(std::initializer_list<T> array) requires (N>0) : DynamicArray(array.begin(), (_validate_size<DIMS...>(array.size()), DIMS)...) {}

    //COPY CONSTRUCTORS
    DynamicArray(const DynamicArray& other) : Base(static_cast<const Base&>(other)), _data((other.size() > 0) ? new T[other.size()] : nullptr) {
        copy_array(this->data(), other.data(), this->size());
    }

    //MOVE CONSTRUCTORS
    DynamicArray(DynamicArray&& other) noexcept : Base(static_cast<Base&&>(std::move(other))), _data(other._data) {
        other._data = nullptr;
    }

    //ASSIGNMENT OPERATORS
    DynamicArray& operator=(const DynamicArray& other) {
        if (&other != this){
            if (this->size() != other.size()){
                delete[] _data;
                _data = other.size() > 0 ? new T[other.size()] : nullptr;                
            }
            Base::operator=(other);
            copy_array(this->data(), other.data(), this->size());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS

    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _data;
            _data = other._data;
            other._data = nullptr;
        }
        return *this;
    }

    ~DynamicArray() {
        delete[] _data;
        _data = nullptr;
    }

    INLINE const T* data() const{
        return _data;
    }

    INLINE T* data() {
        return _data;
    }

    template<IsShapeContainer ShapeContainer>
    void resize(const ShapeContainer& newsize){
        size_t current_size = this->size();
        Base::resize(newsize);//if the new size is invalid, this will throw an error before the execution moves to resizing the _data below.
        size_t total_size = prod(newsize);
        if (total_size != current_size){
            delete[] _data;
            if (total_size == 0){
                _data = nullptr;
            }
            else{
                _data = new T[this->size()];
            }
        }
    }

    template<INT_T... Size>
    void resize(Size... newsize){
        std::array<size_t, sizeof...(newsize)> new_size = {static_cast<size_t>(newsize)...};
        this->resize(new_size);
    }

    T* release(){
        T* res = _data;
        _data = nullptr;
        Base::resize(std::vector<size_t>(this->ndim(), 0));
        return res;
    }

private:

    T* _data = nullptr;

};


template<typename T, Layout L, size_t... DIMS>
class StackArray : public AbstractArray<StackArray<T, L, DIMS...>, T, L, DIMS...>{

    using CLS = StackArray<T, L, DIMS...>;
    using Base = AbstractArray<StackArray<T, L, DIMS...>, T, L, DIMS...>;

public:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

    static_assert(N>0, "StackArray requires only positive template dimensions");

    using Base::Base;

    StackArray() = default;

    explicit StackArray(const T* data) : Base(){
        copy_array<T>(this->data(), data, this->size());
    }

    template<INT_T... Args>
    explicit StackArray(const T* data, Args... shape) : StackArray(shape...){
        copy_array<T>(this->data(), data, this->size());
    }

    template<IsShapeContainer ShapeContainer>
    explicit StackArray(const T* data, const ShapeContainer& shape) : StackArray(shape){
        copy_array<T>(this->data(), data, this->size());
    }

    StackArray(std::initializer_list<T> array) : StackArray(array.begin(), (_validate_size<DIMS...>(array.size()), DIMS)...) {}

    //COPY CONSTRUCTORS
    StackArray(const StackArray& other) : Base(static_cast<const Base&>(other)) {
        copy_array(this->data(), other.data(), this->size());
    }

    //MOVE CONSTRUCTORS
    StackArray(StackArray&& other) noexcept : Base(static_cast<Base&&>(std::move(other))) {
        copy_array(this->data(), other.data(), this->size());
    }

    //ASSIGNMENT OPERATORS
    StackArray& operator=(const StackArray& other) {
        if (&other != this){
            Base::operator=(other);
            copy_array(this->data(), other.data(), this->size());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS
    StackArray& operator=(StackArray&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            copy_array(this->data(), other.data(), this->size());
        }
        return *this;
    }

    ~StackArray() = default;

    INLINE const T* data() const{
        return _data;
    }

    INLINE T* data() {
        return _data;
    }

private:

    T _data[N]{}; //should initialize all values to zero

};


enum class Allocation : std::uint8_t {Heap, Stack, Auto};


template <Allocation Alloc, Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap;

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Heap, L, T, DIMS...> { using type = DynamicArray<T, L, DIMS...>; };

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Stack, L, T, DIMS...> { using type = StackArray<T, L, DIMS...>; };

template <Layout L, typename T>
struct ArrayAllocMap<Allocation::Auto, L, T> { using type = DynamicArray<T, L>; };

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Auto, L, T, DIMS...> {
    using type = std::conditional_t<(((DIMS * ...)*sizeof(T) > 80000) || ((DIMS * ...) == 0)), DynamicArray<T, L, DIMS...>, StackArray<T, L, DIMS...>>; 
};


template <typename T, Allocation Alloc = Allocation::Heap, Layout L = Layout::C, size_t... DIMS>
using Array = ArrayAllocMap<Alloc, L, T, DIMS...>::type;

template <typename T, size_t N=0, Allocation Alloc = Allocation::Heap, Layout L = Layout::C>
using Array1D = ArrayAllocMap<Alloc, L, T, N>::type;

template<typename T, size_t Nr = 0, size_t Nc = 0, Allocation Alloc = Allocation::Auto, Layout L = Layout::C>
class Array2D : public Array<T, Alloc, L, Nr, Nc>{

    using Base = Array<T, Alloc, L, Nr, Nc>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(Array2D)

    INLINE size_t Nrows() const {return this->shape(0);}

    INLINE size_t Ncols() const {return this->shape(1);}

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