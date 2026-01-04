#pragma once

#include "../ndspan.hpp"

template<typename Derived, size_t... DIMS>
class StridedDerivedNdSpan : public DerivedNdSpan<Derived, DIMS...>{

    using Base = DerivedNdSpan<Derived, DIMS...>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(StridedDerivedNdSpan)

    const size_t* strides() const{
        return THIS_C->strides();
    }

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        Derived::set_strides(s, shape, nd);
    }

    static constexpr std::array<size_t, Base::ND> static_strides(){
        std::array<size_t, Base::ND> s{};
        set_strides(s, Base::SHAPE, Base::ND);
        return s;
    }

    template<typename... Idx>
    INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept {
        if constexpr (Base::N > 0) {
            return Derived::strided_unpack(offset, STRIDES, Base::SHAPE, idx...);
        } else {
            return Derived::strided_unpack(offset, this->strides(), this->shape(), idx...);
        }
    }

    inline static constexpr std::array<size_t, Base::ND> STRIDES = static_strides();

protected:

    template<size_t... I, INT_T... Idx>
    INLINE static constexpr size_t _static_offset_impl(std::index_sequence<I...>, Idx... idx) noexcept {
        return ((static_cast<size_t>(idx) * STRIDES[I]) + ...);
    }

};


template<typename Derived, size_t... DIMS>
class StridedStaticNdSpan : public StridedDerivedNdSpan<Derived, DIMS...>{

    using Base = StridedDerivedNdSpan<Derived, DIMS...>;

public:

    using Base::N, Base::ND;
    using Base::Base;

    DEFAULT_RULE_OF_FOUR(StridedStaticNdSpan)
    
    template<INT_T... Idx>
    INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return Base::_static_offset_impl(std::make_index_sequence<Base::ND>(), idx...);
    }

    const size_t* strides() const{
        return Base::STRIDES.data();
    }

};


template<typename Derived, size_t... DIMS>
class StridedSemiStaticNdSpan : public StridedDerivedNdSpan<Derived, DIMS...>{

    using Base = StridedDerivedNdSpan<Derived, DIMS...>;

protected:

    inline static constexpr size_t ND = Base::ND;
    inline static constexpr size_t N = Base::N;

    static_assert(N==0 && ND>0, "StridedSemiStaticNdSpan is for static number of dims of dynamic size");

    StridedSemiStaticNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr StridedSemiStaticNdSpan(Args... shape) : Base(shape...) {
        this->_remake_strides();
    }

    template<IsShapeContainer ShapeContainer>
    explicit constexpr StridedSemiStaticNdSpan(const ShapeContainer& shape) : Base(shape) {
        this->_remake_strides();
    }

    DEFAULT_RULE_OF_FOUR(StridedSemiStaticNdSpan)


public:

    const size_t* strides() const{
        return _fixed_strides;
    }

    template<INT_T... Args>
    void constexpr resize(Args... shape){
        Base::resize(shape...);
        this->_remake_strides();
    }

    template<IsShapeContainer ShapeContainer>
    void constexpr resize(const ShapeContainer& shape){
        Base::resize(shape);
        this->_remake_strides();
    }

    template<INT_T... Idx>
    INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return EXPAND(size_t, ND, I,
            return ((idx * _fixed_strides[I]) + ...);
        );
    }

private:

    inline void _remake_strides(){
        Derived::set_strides(_fixed_strides, this->shape(), this->ndim());
    }
    
    size_t _fixed_strides[ND];

};

template<typename Derived>
class StridedDynamicNdSpan : public StridedDerivedNdSpan<Derived>{

    using Base = StridedDerivedNdSpan<Derived>;
public:
    inline static constexpr size_t ND = 0;
    inline static constexpr size_t N = 0;

protected:

    StridedDynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr StridedDynamicNdSpan(Args... shape) : Base(shape...) {
        _dyn_strides = new size_t[this->ndim()];
        Derived::set_strides(_dyn_strides, this->shape(), this->ndim());
    }

    template<IsShapeContainer ShapeContainer>
    explicit constexpr StridedDynamicNdSpan(const ShapeContainer& shape) : Base(shape) {
        _dyn_strides = shape.size() > 0 ? new size_t[shape.size()] : nullptr;
        Derived::set_strides(_dyn_strides, this->shape(), this->ndim());
    }

    //COPY CONSTRUCTOR
    StridedDynamicNdSpan(const StridedDynamicNdSpan& other) : Base(static_cast<const Base&>(other)){
        _dyn_strides = other.ndim() > 0 ? new size_t[other.ndim()] : nullptr;
        copy_array(_dyn_strides, other._dyn_strides, this->ndim());
    }

    //MOVE CONSTRUCTOR
    StridedDynamicNdSpan(StridedDynamicNdSpan&& other) noexcept : Base(static_cast<Base&&>(other)), _dyn_strides(other._dyn_strides) {
        other._dyn_strides = nullptr;
    }

    //ASSIGNMENT OPERATOR
    StridedDynamicNdSpan& operator=(const StridedDynamicNdSpan& other){
        if (&other != this){
            size_t nd_old = this->ndim();
            Base::operator=(other);
            if (nd_old != other.ndim()){
                delete[] _dyn_strides;
                if (this->ndim() > 0){
                    _dyn_strides = new size_t[this->ndim()];
                }
                else{
                    _dyn_strides = nullptr;
                }
            }
            copy_array(_dyn_strides, other._dyn_strides, this->ndim());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATOR
    StridedDynamicNdSpan& operator=(StridedDynamicNdSpan&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _dyn_strides;
            _dyn_strides = other._dyn_strides;
            other._dyn_strides = nullptr;
        }
        return *this;
    }

    ~StridedDynamicNdSpan() {
        delete[] _dyn_strides;
        _dyn_strides = nullptr;
    }

public:

    const size_t* strides() const{
        return _dyn_strides;
    }

    template<INT_T... Args>
    void constexpr resize(Args... shape){
        size_t nd_old = this->ndim();
        Base::resize(shape...);
        this->_realloc_strides(nd_old);
    }

    template<IsShapeContainer ShapeContainer>
    void constexpr resize(const ShapeContainer& shape){
        size_t nd_old = this->ndim();
        Base::resize(shape);
        this->_realloc_strides(nd_old);
    }



    template<INT_T... Idx>
    INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return EXPAND(size_t, ND, I,
            return ((idx * _dyn_strides[I]) + ...);
        );
    }

private:

    void _realloc_strides(size_t nd_old){
        if (this->ndim() > nd_old){
            //only reallocate in this case
            delete[] _dyn_strides;
            _dyn_strides = new size_t[this->ndim()];
        }
        Derived::set_strides(_dyn_strides, this->shape(), this->ndim());
    }

    size_t* _dyn_strides = nullptr;
};


template<typename Derived, size_t... DIMS>
using StridedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StridedStaticNdSpan<Derived, DIMS...>, std::conditional_t<(sizeof...(DIMS) > 0), StridedSemiStaticNdSpan<Derived, DIMS...>, StridedDynamicNdSpan<Derived>>>;


template<size_t... DIMS>
class RowMajorSpan : public StridedNdSpan<RowMajorSpan<DIMS...>, DIMS...>{

    using Base = StridedNdSpan<RowMajorSpan<DIMS...>, DIMS...>;

public:

    DEFAULT_RULE_OF_FOUR(RowMajorSpan)

    template<IsShapeContainer ShapeContainer>
    explicit RowMajorSpan(const ShapeContainer& shape) : Base(shape) {}

    template<INT_T... Args>
    explicit constexpr RowMajorSpan(Args... shape) : Base(shape...){}

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i=nd; i-- > 0;){
            s[i] = stride;
            stride *= shape[i];
        }
    }

    template<typename STRIDE_T, typename  SHAPE_T, typename... Idx>
    INLINE static void strided_unpack(size_t offset, const STRIDE_T& strides, const SHAPE_T& shape, Idx&... idx) noexcept {
        EXPAND(size_t, sizeof...(idx), I,
            ((idx = offset / strides[I],
            offset %= strides[I]), ...);
        );
    }

};


template<size_t... DIMS>
class ColumnMajorSpan : public StridedNdSpan<ColumnMajorSpan<DIMS...>, DIMS...>{

    using Base = StridedNdSpan<ColumnMajorSpan<DIMS...>, DIMS...>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(ColumnMajorSpan)
    
    template<IsShapeContainer ShapeContainer>
    explicit ColumnMajorSpan(const ShapeContainer& shape) : Base(shape) {}

    template<INT_T... Args>
    explicit constexpr ColumnMajorSpan(Args... shape) : Base(shape...){}

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i = 0; i < nd; ++i) {
            s[i] = stride;
            stride *= shape[i];
        }
    }

    template<typename STRIDE_T, typename  SHAPE_T, typename... Idx>
    INLINE static void strided_unpack(size_t offset, const STRIDE_T& strides, const SHAPE_T& shape, Idx&... idx) noexcept {
        EXPAND(size_t, sizeof...(idx), I,
            ((idx = offset % shape[I],
            offset /= shape[I]), ...);
        );
    }
    
};