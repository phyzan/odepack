#pragma once

#include "ndtools.hpp"

namespace ndspan{

template<typename Derived, size_t... DIMS>
class AbstractNdSpan{

protected:
    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr std::array<size_t, ND> SHAPE = {DIMS...};

    using CLS = AbstractNdSpan<Derived, DIMS...>;

public:
    //ACCESSORS

    INLINE constexpr size_t size() const{
        //override
        return THIS_C->size();
    }

    INLINE constexpr size_t ndim() const {
        //override
        return THIS_C->ndim();
    }

    INLINE const size_t* shape() const {
        //override
        return THIS_C->shape();
    }

    template<INT_T... Idx>
    INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        //override
        return THIS_C->offset_impl(idx...);
    }

    template<INT_T... Idx>
    INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept{
        //override
        return THIS_C->unpack_idx_impl(offset, idx...);
    }

    template<INT_T IDX_T>
    INLINE constexpr size_t shape(IDX_T i) const {
        return shape()[i];
    }

    template<INT_T... Idx>
    INLINE constexpr size_t offset(Idx... idx) const noexcept {
        //dimension and range check in debug mode. They will not be compiled when -DNDEBUG is enabled
        _dim_check(idx...);
        _bounds_check(idx...);
        return offset_impl(idx...);
    }

    template<size_t Nd>
    INLINE constexpr size_t offset(const std::array<size_t, Nd>& idx) const noexcept {
        return EXPAND(size_t, Nd, I,
            return this->offset(idx[I]...);
        );
    }

    template<INT_T... Idx>
    INLINE void unpack_idx(size_t offset, Idx&... idx) const noexcept{
        //dimension and offset check
        _dim_check(idx...);
        _offset_check(offset);
        this->unpack_idx_impl(offset, idx...);
    }

    template<std::integral INT, size_t Nd>
    INLINE void unpack_idx(size_t offset, std::array<INT, Nd>& idx) const noexcept{
        EXPAND(size_t, Nd, I,
            this->unpack_idx(offset, idx[I]...);
        );
    }

    //MODIFIERS

    template<INT_T... Args>
    INLINE void reshape(Args... shape){
        //override
        THIS->reshape(shape...);
    }

    template<INT_T... Args>
    INLINE void constexpr resize(Args... shape){
        //override
        THIS->resize(shape...);
    }

    template<INT_T Int>
    INLINE void reshape(const Int* shape, size_t ndim){
        //override
        THIS->reshape(shape, ndim);
    }

    template<INT_T Int>
    INLINE void resize(const Int* shape, size_t ndim){
        //override
        THIS->resize(shape, ndim);
    }

protected:

    AbstractNdSpan() = default;

    DEFAULT_RULE_OF_FOUR(AbstractNdSpan)

    template<INT_T Int>
    INLINE bool _conserves_shape(const Int* shape, size_t ndim) const{
        if (this->ndim() == ndim && this->ndim() > 0){
            int I = 0;
            return (( shape[I++]==DIMS) && ...);
        }else {
            return false;
        }
    }

    template<INT_T... IntType>
    INLINE static void _shape_check(IntType... dim) {
        if constexpr ((std::is_signed_v<IntType> || ...)) {
            EXPAND(size_t, sizeof...(dim), I,
                assert(((dim >= 0 ) && ...) && "Invalid dims");
            );
        }
    }

    template<INT_T Int>
    INLINE static bool _is_valid_shape(const Int* dims, size_t ndim) {
        for (size_t i=0; i<ndim; i++){
            if (dims[i] < 0){
                return false;
            }
        }
        return true;
    }

    template<INT_T... IntType>
    INLINE void _bounds_check(IntType... idx) const {
        EXPAND(size_t, sizeof...(idx), I,
            assert(((idx >= 0 && size_t(idx) < this->shape(I)) && ...) && "Out of bounds");
        );
    }

    template<INT_T... Idx>
    INLINE void _dim_check(Idx... idx) const {
        //dimension check
        if constexpr (ND > 0){
            static_assert(sizeof...(idx) == ND, "Incorrect number of indices");
        }
        else if constexpr (ND == 0){
            assert(sizeof...(idx) == this->ndim() && "Incorrect number of indices");
        }
    }

    template<std::integral INT>
    INLINE void _offset_check(INT offset) const{
        assert((offset < this->size() && offset >= 0) && "offset is off bounds");
    }

    template<std::integral INT>
    INLINE void _range_check(INT i) const{
        assert((i < this->size() && i >= 0) && "Out of range");
    }

};


template<typename Derived, size_t... DIMS>
class StaticNdSpan : public AbstractNdSpan<Derived, DIMS...>{

    using Base = AbstractNdSpan<Derived, DIMS...>;

protected:
    StaticNdSpan() = default;

    DEFAULT_RULE_OF_FOUR(StaticNdSpan)

    template<INT_T... Args>
    explicit StaticNdSpan(Args... args){
        this->reshape(args...);
    }

    template<INT_T Int>
    explicit StaticNdSpan(const Int* shape, size_t ndim){
        this->reshape(shape, ndim);
    }

public:

    inline constexpr size_t size() const {
        return Base::N;
    }

    inline constexpr size_t ndim() const {
        return Base::ND;
    }

    inline const size_t* shape() const {
        return Base::SHAPE.data();
    }

    using Base::shape;

    template<INT_T... Args>
    INLINE void reshape(Args... shape){
        assert(((shape == DIMS) && ...) && "Runtime dims do not match template dims in reshape");
    }

    template<INT_T... Args>
    INLINE void constexpr resize(Args... shape){
        //can only resize with the exact same shape
        reshape(shape...);
    }

    template<INT_T Int>
    INLINE void reshape(const Int* shape, size_t ndim){
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        assert((ndim == this->ndim()) && "Invalid shape in StaticNdSpan::reshape");
        assert(this->_conserves_shape(shape, ndim) && "Runtime dims do not match template dims in reshape");
    }

    template<INT_T Int>
    INLINE void resize(const Int* shape, size_t ndim){
        reshape(shape, ndim);
    }

};

template<typename Derived, size_t... DIMS>
class SemiStaticNdSpan : public AbstractNdSpan<Derived, DIMS...>{

    using Base = AbstractNdSpan<Derived, DIMS...>;
    using CLS = SemiStaticNdSpan<Derived, DIMS...>;

protected:

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (DIMS * ...);

    static_assert(N==0 && ND>0, "Use SemiStaticNdSpan for a static number of dims of dynamic length");

    SemiStaticNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr SemiStaticNdSpan(Args... shape) {
        this->resize(shape...);
    }

    template<INT_T Int>
    explicit SemiStaticNdSpan(const Int* shape, size_t ndim){
        this->resize(shape, ndim);
    }

    DEFAULT_RULE_OF_FOUR(SemiStaticNdSpan)

public:

    template<INT_T... Args>
    void constexpr resize(Args... shape){
        //ND > 0, but some template dims are zero
        static_assert((sizeof...(shape) == ND), "Constructor must be called with as many dims as the number of template dims");

        _data[0] = size_t((shape*...));
        EXPAND(size_t, ND, I,
            assert(((shape >= 0 && (Base::ND==0 || ((Base::SHAPE[I] > 0 ? shape == Base::SHAPE[I] : true)))) && ...) && "Runtime dims do not match template dims");
            ((_data[I+1] = size_t(shape)), ...);
        );
    }

    template<INT_T Int>
    void resize(const Int* shape, size_t ndim){
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        //TODO make sure non of these are zero at runtime, and they are equal to the template parameter
        assert(ndim == Base::ND && "SemiStaticNdSpan::resize invalid shape");

        _data[0] = prod(shape, ndim);
        copy_array(_data+1, shape, ndim);
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        assert(((shape*...) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape...);
    }

    template<INT_T Int>
    void reshape(const Int* shape, size_t ndim){
        assert((prod(shape, ndim) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape, ndim);
    }

    INLINE size_t size() const{
        return _data[0];
    }

    INLINE constexpr size_t ndim() const{
        return ND;
    }

    INLINE const size_t* shape() const {
        return _data+1;
    }

    using Base::shape;

private:

    size_t _data[ND+1] = {DIMS...};


};

class SemiStaticSpan1D : public AbstractNdSpan<SemiStaticSpan1D, 0>{

    using Base = AbstractNdSpan<SemiStaticSpan1D, 0>;
    using CLS = SemiStaticSpan1D;

public:

    inline static constexpr size_t ND = 1;
    inline static constexpr size_t N = 0;

    SemiStaticSpan1D() = default;

    template<INT_T Int>
    explicit constexpr SemiStaticSpan1D(Int size) {
        this->resize(size);
    }

    template<INT_T Int>
    explicit SemiStaticSpan1D(const Int* shape, size_t ndim){
        this->resize(shape, ndim);
    }

    DEFAULT_RULE_OF_FOUR(SemiStaticSpan1D)

    template<INT_T Int>
    inline void constexpr resize(Int newsize){
        _size = newsize;
    }

    template<INT_T Int>
    void resize(const Int* shape, size_t ndim){
        //TODO make sure they are equal to the non zero template dims
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");      
        assert(ndim == 1 && "SemiStaticSpan1D::resize invalid shape");
        _size = size_t(shape[0]);
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        assert(((shape*...) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape...);
    }

    template<INT_T Int>
    void reshape(const Int* shape, size_t ndim){
        assert((prod(shape, ndim) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape, ndim);
    }

    INLINE size_t size() const{
        return _size;
    }

    INLINE constexpr size_t ndim() const{
        return ND;
    }

    INLINE const size_t* shape() const {
        return &_size;
    }


    template<INT_T Int>
    INLINE constexpr size_t offset_impl(Int idx) const noexcept{
        return idx;
    }

    template<INT_T Int>
    INLINE void unpack_idx_impl(size_t offset, Int& idx) const noexcept{
        idx=offset;
    }

    using Base::shape;

private:

    size_t _size = 0;


};

template<typename Derived>
class DynamicNdSpan : public AbstractNdSpan<Derived>{

    using Base = AbstractNdSpan<Derived>;
    using CLS = DynamicNdSpan<Derived>;

    static constexpr size_t DEFAULT_DATA[2] = {0, 0};

protected:

    static constexpr size_t ND = 0;
    static constexpr size_t N = 0;
    

    static_assert(N==0, "For fixed compile time dims, use StaticNdSpan");

    DynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr DynamicNdSpan(Args... shape){
        Base::_shape_check(shape...);
        _data = new size_t[2+sizeof...(shape)]{};
        this->resize(shape...);
    }

    template<INT_T Int>
    explicit DynamicNdSpan(const Int* shape, size_t ndim) {
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        _data = new size_t[2+ndim](0);
        this->resize(shape, ndim);
    }

    //COPY CONSTRUCTOR
    DynamicNdSpan(const DynamicNdSpan& other) : Base(other){
        if (other.ndim() > 0){
            _data = new size_t[2+other.ndim()];
            ptr()[0] = other.ndim();
            ptr()[1] = other.size();
            copy_array(ptr()+2, other.shape(), this->ndim());
        }
    }

    //MOVE CONSTRUCTOR
    DynamicNdSpan(DynamicNdSpan&& other) noexcept : Base(other), _data(other._data) {
        other._data = DEFAULT_DATA;
    }

    //ASSIGNMENT OPERATOR
    DynamicNdSpan& operator=(const DynamicNdSpan& other) {
        if (&other != this){
            Base::operator=(other);
            if (this->ndim() != other.ndim()){
                if (this->ndim() > 0){
                    delete[] ptr();
                }
                if (other.ndim() > 0){
                    _data = new size_t[2+other.ndim()]{other.ndim(), other.size()};
                }else{
                    _data = DEFAULT_DATA;
                }
            }
            if (other.ndim() > 0){
                copy_array(ptr()+2, other.shape(), other.ndim());
            }
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATOR
    DynamicNdSpan& operator=(DynamicNdSpan&& other) noexcept{
        if (&other != this){
            Base::operator=(std::move(other));
            _data = other._data;
            other._data = DEFAULT_DATA;
        }
        return *this;
    }

public:

    template<INT_T... Args>
    void constexpr resize(Args... shape){

        constexpr size_t new_nd = sizeof...(shape);
        if constexpr (new_nd == 0) {
            if (this->ndim() > 0){
                delete[] ptr();
                _data = DEFAULT_DATA;
            }
        }else{
            if (new_nd != this->ndim()){
                if (this->ndim() > 0){
                    delete[] ptr();
                }
                _data = new size_t[2+new_nd];
                ptr()[0] = new_nd;
            }

            ptr()[1] = (shape*...);

            EXPAND(size_t, new_nd, I,
                if constexpr ((std::is_signed_v<Args> || ...)) {
                    assert(((shape >= 0 ) && ...) && "Invalid dims");
                }
                ((ptr()[I+2] = shape), ...);
            );
        }
        
    }

    template<INT_T Int>
    void resize(const Int* shape, size_t ndim){

        assert(Base::_is_valid_shape(shape, ndim) && "Negative dims not allowed");

        if (ndim == 0) {
            if (this->ndim() > 0){
                delete[] ptr();
                _data = DEFAULT_DATA;
            }
        }else{
            if (ndim != this->ndim()){
                if (this->ndim() > 0){
                    delete[] ptr();
                }
                _data = new size_t[2+ndim];

                ptr()[0] = ndim;
            }
            ptr()[1] = prod(shape, ndim);

            copy_array(ptr()+2, shape, ndim);
        }
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        assert((size_t((shape*...)) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape...);
    }

    template<INT_T Int>
    void reshape(const Int* shape, size_t ndim){
        if (prod(shape, ndim) != this->size()){
            throw std::runtime_error("Invalid new shape. The total size of the array is not conserved");
        }
        this->resize(shape, ndim);
    }

    INLINE size_t ndim() const{
        return _data[0];
    }
    
    INLINE size_t size() const{
        return _data[1];
    }

    INLINE const size_t* shape() const {
        return this->ndim() > 0 ? _data+2 : nullptr;
    }

    using Base::shape;

protected:

    ~DynamicNdSpan() {
        if (this->ndim() > 0){
            delete[] ptr();
        }
    }

private:

    inline size_t*& ptr(){
        return const_cast<size_t*&>(_data);
    }

    const size_t* _data = DEFAULT_DATA; //size is number of dims+2: [nd, n, dims...]


};


// Derived Layouts inherit from this
template<typename Derived, size_t... DIMS>
using DerivedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StaticNdSpan<Derived, DIMS...>, std::conditional_t<(sizeof...(DIMS)>0), SemiStaticNdSpan<Derived, DIMS...>, DynamicNdSpan<Derived>>>;

} // namespace ndspan