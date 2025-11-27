#pragma once

#include "ndtools.hpp"

template<typename Derived, size_t... DIMS>
struct AbstractNdSpan{

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr std::array<size_t, ND> SHAPE = {DIMS...};

    using CLS = AbstractNdSpan<Derived, DIMS...>;

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
        //dimension and range check in debug mode. They will not be compiled when -DNDBUG is enabled
        _dim_check(idx...);
        _bounds_check(std::make_index_sequence<sizeof...(idx)>(), idx...);
        return offset_impl(idx...);
    }

    template<size_t Nd>
    INLINE constexpr size_t offset_from_array(const std::array<size_t, Nd>& idx) const noexcept {
        return _offset_from_array_aux(idx, std::make_index_sequence<Nd>());
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
        return this->_unpack_idx_aux(offset, idx, std::make_index_sequence<Nd>());
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

    template<IsShapeContainer ShapeContainer>
    INLINE void reshape(const ShapeContainer& shape){
        //override
        THIS->reshape(shape);
    }

    template<IsShapeContainer ShapeContainer>
    INLINE void resize(const ShapeContainer& shape){
        //override
        THIS->resize(shape);
    }

protected:

    AbstractNdSpan() = default;

    template<INT_T... ARGS, size_t... I>
    INLINE void _bounds_check(std::index_sequence<I...>, ARGS... idx) const {
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wsign-compare"
        assert(((idx >= 0 && idx < this->shape(I)) && ...) && "Out of bounds");
        #pragma GCC diagnostic pop
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

private:

    template<std::integral INT, size_t Nd, size_t... I>
    INLINE void _unpack_idx_aux(size_t offset, std::array<INT, Nd>& idx, std::index_sequence<I...>) const {
        return this->unpack_idx(offset, idx[I]...);
    }

    template<size_t Nd, size_t... I>
    INLINE constexpr size_t _offset_from_array_aux(const std::array<size_t, Nd>& idx, std::index_sequence<I...>) const noexcept {
        return this->offset(idx[I]...);
    }

};


template<typename Derived, size_t... DIMS>
struct StaticNdSpan : public AbstractNdSpan<Derived, DIMS...>{

    using Base = AbstractNdSpan<Derived, DIMS...>;

    StaticNdSpan() = default;

    DEFAULT_RULE_OF_FOUR(StaticNdSpan)

    template<INT_T... Args>
    explicit StaticNdSpan(Args... args){
        this->reshape(args...);
    }

    template<IsShapeContainer ShapeContainer>
    explicit StaticNdSpan(const ShapeContainer& shape){
        this->reshape(shape);
    }

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

    template<IsShapeContainer ShapeContainer>
    INLINE void reshape(const ShapeContainer& shape){
        assert((shape.size() == this->ndim()) && "Invalid shape in StaticNdSpan::reshape");
        assert([&]{int I = 0; return ((shape[I++] == DIMS) && ...);}() &&  "Runtime dims do not match template dims in reshape");}

    template<IsShapeContainer ShapeContainer>
    INLINE void resize(const ShapeContainer& shape){
        reshape(shape);
    }

};



template<typename Derived, size_t... DIMS>
class DynamicNdSpan : public AbstractNdSpan<Derived, DIMS...>{

    using Base = AbstractNdSpan<Derived, DIMS...>;
    using CLS = DynamicNdSpan<Derived, DIMS...>;

protected:

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));

    static_assert(N==0, "For fixed compile time dims, use StaticNdSpan");

public:

    DynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr DynamicNdSpan(Args... shape) : _n(sizeof...(shape) == 0 ? 0 : (shape*...*1)) {
        this->reshape(shape...);
    }

    template<IsShapeContainer ShapeContainer>
    explicit DynamicNdSpan(const ShapeContainer& shape){
        this->resize(shape);
    }

    //COPY CONSTRUCTORS

    DynamicNdSpan(const DynamicNdSpan& other) requires (ND>0) = default;

    DynamicNdSpan(const DynamicNdSpan& other) requires (ND==0) : Base(other), _nd(other._nd), _n(other._n){
        _dyn_shape = other._nd > 0 ? new size_t[other._nd] : nullptr;
        copy_array(_dyn_shape, other._dyn_shape, _nd);
    }

    //MOVE CONSTRUCTORS
    DynamicNdSpan(DynamicNdSpan&& other) requires (ND>0) = default;

    DynamicNdSpan(DynamicNdSpan&& other) noexcept requires (ND==0) : Base(other), _nd(other._nd), _n(other._n), _dyn_shape(other._dyn_shape) {
        other._dyn_shape = nullptr;
    }

    //ASSIGNMENT OPERATORS
    DynamicNdSpan& operator=(const DynamicNdSpan&) requires (ND>0) = default;

    DynamicNdSpan& operator=(const DynamicNdSpan& other) requires (ND==0){
        if (&other != this){
            Base::operator=(other);
            if (_nd != other._nd){
                _nd = other._nd;
                delete[] _dyn_shape;
                _dyn_shape = (_nd > 0 ? new size_t[_nd] : nullptr);
            }
            _n = other._n;
            copy_array(_dyn_shape, other._dyn_shape, _nd);
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS
    DynamicNdSpan& operator=(DynamicNdSpan&&) requires (ND>0) = default;

    DynamicNdSpan& operator=(DynamicNdSpan&& other) noexcept requires (ND==0){
        if (&other != this){
            Base::operator=(std::move(other));
            _n = other._n;
            _nd = other._nd;
            delete[] _dyn_shape;
            _dyn_shape = other._dyn_shape;
            other._n = 0;
            other._nd = 0;
            other._dyn_shape = nullptr;
        }
        return *this;
    }

    ~DynamicNdSpan() {
        if constexpr (ND==0){
            delete[] _dyn_shape;
            _dyn_shape = nullptr;
        }
    }

    template<INT_T... Args>
    void constexpr resize(Args... shape){

        constexpr size_t new_nd = sizeof...(shape);
        static_assert(new_nd > 0, "Cannot call resize() with no arguments");

        if constexpr (ND==0){
            if (new_nd != _nd){
                if (new_nd > _nd){
                    //only reallocate in this case
                    delete[] _dyn_shape;
                    _dyn_shape = new size_t[new_nd];
                }
                _nd = new_nd;
            }
            _set_shape(_dyn_shape, std::make_index_sequence<sizeof...(shape)>(), shape...);
        }
        else {
            //ND > 0, but some template dims are zero
            static_assert((new_nd == ND), "Constructor must be called with as many dims as the number of template dims");
            _set_shape(_fixed_shape, std::make_index_sequence<sizeof...(shape)>(), shape...);
        }

        _n = (shape*...);
    }


    template<IsShapeContainer ShapeContainer>
    void constexpr resize(const ShapeContainer& shape){
        assert_integral_data(shape);
        //TODO make sure non of these are zero at runtime, and they are equal to the template parameter
        size_t new_nd = shape.size();
        if (new_nd == 0){
            throw std::runtime_error("Cannot call resize() with no arguments");
        }
        else if (Base::ND != 0 && new_nd != Base::ND){
            throw std::runtime_error("DynamicNdSpan::resize invalid shape");
        }

        if constexpr (ND==0){
            if (new_nd != _nd){
                if (new_nd > _nd){
                    //only reallocate in this case
                    delete[] _dyn_shape;
                    _dyn_shape = new size_t[new_nd];
                }
                _nd = new_nd;
            }
            copy_array(_dyn_shape, shape.data(), new_nd);
        }
        else {
            //ND > 0, but some template dims are zero
            for (size_t i=0; i<new_nd; i++){
                if (Base::SHAPE[i] > 0 && shape[i] != Base::SHAPE[i]){
                    throw std::runtime_error("Runtime dims in do not match template dims");
                }
            }
            copy_array(_fixed_shape, shape.data(), new_nd);
        }

        _n = prod(shape);
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        assert(((shape*...) == _n) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape...);
    }

    template<IsShapeContainer ShapeContainer>
    void constexpr reshape(const ShapeContainer& shape){
        if (prod(shape) != _n){
            throw std::runtime_error("Invalid new shape. The total size of the array is not conserved");
        }
        this->resize(shape);
    }

    INLINE size_t size() const{
        return _n;
    }

    INLINE size_t ndim() const{
        if constexpr (ND>0){
            return ND;
        }
        else{
            return _nd;
        }
    }

    INLINE const size_t* shape() const {
        if constexpr (ND>0){
            return _fixed_shape;
        }
        else{
            return _dyn_shape;
        }
    }

    using Base::shape;

private:


    template<INT_T... ARGS, size_t... I>
    INLINE void _set_shape(size_t* my_shape, std::index_sequence<I...>, ARGS... shape){
        static_assert(Base::ND==0 || (Base::ND == sizeof...(shape)), "Invalid number of dims");
        assert(((shape >= 0 && (Base::ND==0 || ((Base::SHAPE[I] > 0 ? shape == Base::SHAPE[I] : true)))) && ...) && "Runtime dims do not match template dims");
        ((my_shape[I] = shape), ...);
    }

    size_t  _nd = ND;
    size_t  _n = 0;
    size_t* _dyn_shape = nullptr;
    size_t  _fixed_shape[ND>0 ? ND : 1] = {DIMS...}; //used when ND > 0

};


// Derived Layouts inherit from this
template<typename Derived, size_t... DIMS>
using DerivedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StaticNdSpan<Derived, DIMS...>, DynamicNdSpan<Derived, DIMS...>>;
