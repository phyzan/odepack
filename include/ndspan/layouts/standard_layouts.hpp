#pragma once

#include "../ndspan.hpp"

template<typename DerivedMajor, typename T, size_t... DIMS>
class StridedDerivedNdSpan : public DerivedNdSpan<StridedDerivedNdSpan<DerivedMajor, T, DIMS...>, T, DIMS...>{

    using Base = DerivedNdSpan<StridedDerivedNdSpan<DerivedMajor, T, DIMS...>, T, DIMS...>;

public:

    template<INT_T... Idx>
    inline constexpr size_t offset_impl(Idx... idx) const noexcept{
        return static_cast<const DerivedMajor*>(this)->offset_impl(idx...);
    }

protected:

    using Base::Base;

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        DerivedMajor::set_strides(s, shape, nd);
    }

    static constexpr std::array<size_t, Base::ND> static_strides(){
        std::array<size_t, Base::ND> s{};
        set_strides(s, Base::SHAPE, Base::ND);
        return s;
    }

    inline static constexpr std::array<size_t, Base::ND> STRIDES = static_strides();

    template<size_t... I, INT_T... Idx>
    inline constexpr size_t _static_offset_impl(std::index_sequence<I...>, Idx... idx) const noexcept {
        return ((static_cast<size_t>(idx) * STRIDES[I]) + ...);
    }

};


template<typename DerivedMajor, typename T, size_t... DIMS>
class StridedStaticNdSpan : public StridedDerivedNdSpan<DerivedMajor, T, DIMS...>{

    using Base = StridedDerivedNdSpan<DerivedMajor, T, DIMS...>;

    using Base::Base;

public:
    template<INT_T... Idx>
    inline constexpr size_t offset_impl(Idx... idx) const noexcept{
        return Base::_static_offset_impl(std::make_index_sequence<Base::ND>(), idx...);
    }

};


template<typename DerivedMajor, typename T, size_t... DIMS>
class StridedDynamicNdSpan : public StridedDerivedNdSpan<DerivedMajor, T, DIMS...>{

    using Base = StridedDerivedNdSpan<DerivedMajor, T, DIMS...>;
    inline static constexpr size_t ND = Base::ND;
    inline static constexpr size_t N = Base::N;

protected:

    StridedDynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr StridedDynamicNdSpan(Args... shape) : Base(shape...) {
        if constexpr (ND == 0){
            _dyn_strides = new size_t[sizeof...(shape)];
            Base::set_strides(_dyn_strides, this->shape(), this->ndim());
        }
        else if constexpr (N==0){
            Base::set_strides(_fixed_strides, this->shape(), this->ndim());
        }
    }

    //COPY CONSTRUCTORS

    StridedDynamicNdSpan(const StridedDynamicNdSpan& other) requires (ND>0) = default;

    StridedDynamicNdSpan(const StridedDynamicNdSpan& other) requires (ND==0) : Base(other){
        _dyn_strides = other.ndim() > 0 ? new size_t[other.ndim()] : nullptr;
        copy_array(_dyn_strides, other._dyn_strides, this->ndim());
    }

    //MOVE CONSTRUCTORS
    StridedDynamicNdSpan(StridedDynamicNdSpan&& other) requires (ND>0) = default;

    StridedDynamicNdSpan(StridedDynamicNdSpan&& other) noexcept requires (ND==0) : Base(other), _dyn_strides(other._dyn_strides) {
        other._dyn_strides = nullptr;
    }

    //ASSIGNMENT OPERATORS
    StridedDynamicNdSpan& operator=(const StridedDynamicNdSpan&) requires (ND>0) = default;

    StridedDynamicNdSpan& operator=(const StridedDynamicNdSpan& other) requires (ND==0){
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

    //MOVE-ASSIGNMENT OPERATORS
    StridedDynamicNdSpan& operator=(StridedDynamicNdSpan&&) requires (ND>0) = default;

    StridedDynamicNdSpan& operator=(StridedDynamicNdSpan&& other) noexcept requires (ND==0){
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _dyn_strides;
            _dyn_strides = other._dyn_strides;
            other._dyn_strides = nullptr;
        }
        return *this;
    }

    ~StridedDynamicNdSpan() {
        if constexpr (ND==0){
            delete[] _dyn_strides;
            _dyn_strides = nullptr;
        }
    }

public:

    template<INT_T... Args>
    void constexpr resize(Args... shape){
        size_t nd_old = this->ndim();
        Base::resize(shape...);
        if constexpr (ND == 0){
            if (this->ndim() > nd_old){
                //only reallocate in this case
                delete[] _dyn_strides;
                _dyn_strides = new size_t[this->ndim()];
            }
            Base::set_strides(_dyn_strides, this->shape(), this->ndim());
        }
        else if constexpr (N == 0){
            Base::set_strides(_fixed_strides, this->shape(), this->ndim());
        }
    }

    template<INT_T... Idx>
    inline constexpr size_t offset_impl(Idx... idx) const noexcept{
        if constexpr (N > 0){
            return Base::offset_impl(idx...);
        }
        return _dynamic_offset_impl(std::make_index_sequence<sizeof...(idx)>(), idx...);
    }

protected:

    template<size_t... I, INT_T... Idx>
    inline constexpr size_t _dynamic_offset_impl(std::index_sequence<I...>, Idx... idx) const noexcept {
        if constexpr (ND == 0){
            return ((static_cast<size_t>(idx) * _dyn_strides[I]) + ...);
        }
        return ((static_cast<size_t>(idx) * _fixed_strides[I]) + ...);
    }

    size_t* _dyn_strides = nullptr;
    std::array<size_t, ND> _fixed_strides = Base::static_strides();//used when ND > 0

};


template<typename DerivedMajor, typename T, size_t... DIMS>
using StridedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StridedStaticNdSpan<DerivedMajor, T, DIMS...>, StridedDynamicNdSpan<DerivedMajor, T, DIMS...>>;


template<typename T, size_t... DIMS>
class RowMajorSpan : public StridedNdSpan<RowMajorSpan<T, DIMS...>, T, DIMS...>{

    using Base = StridedNdSpan<RowMajorSpan<T, DIMS...>, T, DIMS...>;

public:

    using Base::Base;

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i=nd; i-- > 0;){
            s[i] = stride;
            stride *= shape[i];
        }
    }

};


template<typename T, size_t... DIMS>
class ColumnMajorSpan : public StridedNdSpan<ColumnMajorSpan<T, DIMS...>, T, DIMS...>{

    using Base = StridedNdSpan<ColumnMajorSpan<T, DIMS...>, T, DIMS...>;

public:

    using Base::Base;

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i = 0; i < nd; ++i) {
            s[i] = stride;
            stride *= shape[i];
        }
    }
    
};