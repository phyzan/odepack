#pragma once

#include "ndview.hpp"

namespace ndspan{


template<typename Derived, typename T, Layout L, size_t... DIMS>
class AbstractArray : public AbstractMutView<Derived, L, T, DIMS...>{

    using CLS = AbstractArray<Derived, T, L, DIMS...>;
    using Base = AbstractMutView<Derived, L, T, DIMS...>;

protected:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

    using Base::Base;

    AbstractArray() = default;

    DEFAULT_RULE_OF_FOUR(AbstractArray)

    template<INT_T Int>
    explicit AbstractArray(const Int* shape, size_t ndim) : Base(shape, ndim) {}

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

    template<INT_T Int>
    explicit DynamicArray(const Int* shape, size_t ndim) : Base(shape, ndim){
        if (this->size() > 0){
            _data = new T[this->size()];
        }
    }

    template<INT_T Int>
    explicit DynamicArray(const T* data, const Int* shape, size_t ndim) : DynamicArray(shape, ndim){
        copy_array<T>(this->data(), data, this->size());
    }

    template<INT_T Int>
    explicit DynamicArray(T* data, const Int* shape, size_t ndim, bool own_it = false) : Base(shape, ndim){
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

    //COPY CONSTRUCTOR
    DynamicArray(const DynamicArray& other) : Base(static_cast<const Base&>(other)), _data((other.size() > 0) ? new T[other.size()] : nullptr) {
        copy_array(this->data(), other.data(), this->size());
    }

    //MOVE CONSTRUCTOR
    DynamicArray(DynamicArray&& other) noexcept : Base(static_cast<Base&&>(std::move(other))), _data(other.release()) {}

    //ASSIGNMENT OPERATOR
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

    //MOVE-ASSIGNMENT OPERATOR
    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _data;
            _data = other.release();
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

    template<INT_T Int>
    void resize(const Int* newsize, size_t ndim){
        size_t current_size = this->size();
        Base::resize(newsize, ndim);//if the new size is invalid, this will throw an error before the execution moves to resizing the _data below.
        size_t total_size = prod(newsize, ndim);
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
        this->resize(new_size.data(), new_size.size());
    }

    T* release(){
        T* res = _data;
        _data = nullptr;
        _reset_base_to_zero(std::make_index_sequence<Base::ND>());
        return res;
    }

private:

    template<size_t... I>
    INLINE void _reset_base_to_zero(std::index_sequence<I...>){
        if constexpr (Base::N == 0) {
            Base::resize((Base::SHAPE[I])...);
        }
    }

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

    template<INT_T Int>
    explicit StackArray(const T* data, const Int* shape, size_t ndim) : StackArray(shape, ndim){
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
class Array : public ArrayAllocMap<Alloc, L, T, DIMS...>::type{

    using Base = ArrayAllocMap<Alloc, L, T, DIMS...>::type;

public:

    using value_type = Base::value_type;
    using iterator = Base::iterator;
    using const_iterator = Base::const_iterator;

    using Base::Base;

    //=============================== ACCESSORS ===================================

    // NdSpan interface
    INLINE constexpr size_t size() const{
        return Base::size();
    }

    INLINE constexpr size_t ndim() const {
        return Base::ndim();
    }

    INLINE const size_t* shape() const {
        return Base::shape();
    }

    template<INT_T IDX_T>
    INLINE constexpr size_t shape(IDX_T i) const {
        return Base::shape(i);
    }

    template<INT_T... Idx>
    INLINE constexpr size_t offset(Idx... idx) const noexcept {
        return Base::offset(idx...);
    }

    template<size_t Nd>
    INLINE constexpr size_t offset(const std::array<size_t, Nd>& idx) const noexcept {
        return Base::offset(idx);
    }

    template<INT_T... Idx>
    INLINE void unpack_idx(size_t offset, Idx&... idx) const noexcept{
        Base::unpack_idx(offset, idx...);
    }

    template<std::integral INT, size_t Nd>
    INLINE void unpack_idx(size_t offset, std::array<INT, Nd>& idx) const noexcept{
        Base::unpack_idx(offset, idx);
    }

    // View interface
    const_iterator begin() const { return Base::begin(); }
    const_iterator end() const { return Base::end(); }

    INLINE const T* data() const{
        return Base::data();
    }

    template<INT_T... Int>
    INLINE const T* ptr(Int... idx) const{
        return Base::ptr(idx...);
    }

    INLINE View<T, L, DIMS...> view() const{
        if constexpr (Base::N > 0){
            return View<T, L, DIMS...>(this->data());
        }else if (Base::ND > 0) {
            const size_t* s = this->shape();
            return EXPAND(size_t, Base::ND, I,
                View<T, L, DIMS...>(this->data(), s[I]...);
            );
        }else {
            return View<T, L, DIMS...>(this->data(), this->shape(), this->ndim());
        }
    }

    template<INT_T... Idx>
    INLINE const T& operator()(Idx... idx) const {
        return Base::operator()(idx...);
    }

    template<typename... Idx>
    INLINE auto operator()(Idx... i) const{
        return Base::operator()(i...);
    }

    template<INT_T IDX_T>
    INLINE const T& operator[](IDX_T i) const{
        return Base::operator[](i);
    }


    //=============================== MODIFIERS ===================================

    // NdSpan interface
    template<INT_T... Args>
    INLINE void reshape(Args... shape){
        Base::reshape(shape...);
    }

    template<INT_T... Args>
    INLINE void constexpr resize(Args... shape){
        Base::resize(shape...);
    }

    template<INT_T Int>
    INLINE void reshape(const Int* shape, size_t ndim){
        Base::reshape(shape, ndim);
    }

    template<INT_T Int>
    INLINE void resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim);
    }


    // MutView interface

    iterator begin() { return Base::data(); }
    iterator end()   { return Base::end(); }

    INLINE T* data(){
        return Base::data();
    }

    template<INT_T... Idx>
    INLINE T& operator()(Idx... idx) {
        return Base::operator()(idx...);
    }

    template<typename... Idx>
    INLINE auto operator()(Idx... i){
        return Base::operator()(i...);
    }

    template<INT_T IDX_T>
    INLINE T& operator[](IDX_T i){
        return Base::operator[](i);
    }

    template<INT_T... Int>
    INLINE T* ptr(Int... idx){
        return Base::ptr(idx...);
    }

    INLINE Array& set(const T& value){
        Base::set(value);
        return *this;
    }

};




template <typename T, size_t NDIM, Allocation Alloc = Allocation::Heap, Layout L = Layout::C>
struct HelperNdArray
{
private:
    template <std::size_t... Is>
    static Array<T, Alloc, L, (static_cast<void>(Is), 0)...> make(std::index_sequence<Is...>);

public:

    using type = decltype(make(std::make_index_sequence<NDIM>{}));
};

template <typename T, size_t NDIM, Allocation Alloc = Allocation::Heap, Layout L = Layout::C>
using NdArray = HelperNdArray<T, NDIM, Alloc, L>::type;


template <typename T, size_t SIZE=0, Allocation Alloc = Allocation::Heap>
using Array1D = Array<T, Alloc, Layout::C, SIZE>;

template<typename T, size_t Nr = 0, size_t Nc = 0, Allocation Alloc = Allocation::Heap, Layout L = Layout::C>
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


template <typename T, size_t M=0, size_t N=0, size_t K=0, Allocation Alloc = Allocation::Heap>
using Array3D = Array<T, Alloc, Layout::C, M, N, K>;

} // namespace ndspan