#pragma once

#include "ndtools.hpp"

template<typename DerivedLayout, typename T, size_t... DIMS>
struct BaseNdSpan{

// static_assert((L != Layout::Z) || (... && (((DIMS & (DIMS - 1)) == 0))), "Z-order layout supported only for dimensions that are a power of 2");

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t SHAPE[ND > 0 ? ND : 1] = {DIMS...};

    BaseNdSpan() = default;

    template<INT_T... Args>
    explicit BaseNdSpan(Args... args){
        this->reshape(args...);
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const noexcept {
        return static_cast<const DerivedLayout*>(this)->offset_impl(idx...);
    }

    inline static constexpr size_t size(){
        return N;
    }

    inline static constexpr size_t ndim(){
        return ND;
    }

    inline static const size_t* shape() {
        return SHAPE;
    }

    inline static constexpr size_t shape(size_t i) {
        return SHAPE[i];
    }

    template<INT_T... Args>
    inline void reshape(Args... shape){
        if (! ((shape==DIMS) && ...)){
            throw std::runtime_error("Runtime dims do not match template dims in Tensor constructor");
        }
    }

    template<INT_T... Args>
    inline void constexpr resize(Args... shape){
        //can only resize with the exact same shape
        reshape(shape...);
    }

};



template<typename DerivedLayout, typename T, size_t... DIMS>
class DynamicNdSpan : public BaseNdSpan<DerivedLayout, T, DIMS...>{

    using Base = BaseNdSpan<DerivedLayout, T, DIMS...>;
    using CLS = DynamicNdSpan<DerivedLayout, T, DIMS...>;

protected:

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));

public:

    DynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr DynamicNdSpan(Args... shape) : _n(sizeof...(shape) == 0 ? 0 : (shape*...*1)) {
        this->reshape(shape...);
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

        //TODO make sure non of these are zero at runtime, and they are equal to the template parameter
        constexpr size_t new_nd = sizeof...(shape);
        size_t new_dims[new_nd] = {static_cast<size_t>(shape)...};
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
            copy_array(_dyn_shape, new_dims, _nd);
        }
        else if constexpr (N==0){
            //ND > 0, but some template dims are zero
            static_assert((new_nd == ND), "Constructor must be called with as many dims as the number of template dims");
            for (size_t i=0; i<ND; i++){
                if ((Base::SHAPE[i] > 0) && new_dims[i] != Base::SHAPE[i]){
                    throw std::runtime_error("Runtime dims do not match template dims");
                }
                _fixed_shape[i] = new_dims[i];
            }
        }
        else{
            Base::reshape(new_dims);
        }

        if constexpr (N==0) {
            _n = (shape*...);
        }
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        const size_t new_size = (shape*...);
        if (new_size != _n){
            throw std::runtime_error("Invalid new shape. The total size of the tensor is not conserved");
        }
        return static_cast<DerivedLayout*>(this)->resize(shape...);
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const {
        if constexpr (ND > 0){
            STATIC_IDX_ASSERT(idx, ND);
        }
        else if constexpr (ND == 0){
            DYNAMIC_IDX_ASSERT(idx, this->ndim());
        }
        return Base::offset(idx...);
    }

    inline size_t size() const{
        if constexpr (N > 0) {return N;}
        else {return _n;}
    }

    inline size_t ndim() const{
        if constexpr (ND>0){
            return ND;
        }
        else{
            return _nd;
        }
    }

    inline const size_t* shape() const {
        if constexpr (ND>0){
            return _fixed_shape;
        }
        else{
            return _dyn_shape;
        }
    }

    inline size_t shape(size_t i) const {
        if constexpr (ND>0){
            return _fixed_shape[i];
        }
        else{
            return _dyn_shape[i];
        }
    }

    size_t  _nd = ND;
    size_t  _n = (ND == 0 ? 0 : (DIMS * ... * 1));
    size_t* _dyn_shape = nullptr;
    size_t  _fixed_shape[ND>0 ? ND : 1] = {DIMS...}; //used when ND > 0, but N==0

};


// Derived Layouts inherit from this
template<typename DerivedLayout, typename T, size_t... DIMS>
using DerivedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), BaseNdSpan<DerivedLayout, T, DIMS...>, DynamicNdSpan<DerivedLayout, T, DIMS...>>;
