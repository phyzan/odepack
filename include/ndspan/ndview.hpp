#pragma once

#include "layoutmap.hpp"

template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractNdView : public NdSpan<L, DIMS...>{

    using CLS = AbstractNdView<Derived, L, T, DIMS...>;
    using Base = NdSpan<L, DIMS...>;

protected:

    using Base::Base;
    DEFAULT_RULE_OF_FOUR(AbstractNdView)

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    //ACCESSORS

    iterator begin() { return this->data(); }
    iterator end()   { return this->data() + this->size(); }

    const_iterator begin() const { return this->data(); }
    const_iterator end() const { return this->data() + this->size(); }
    
    INLINE const T* data() const{
        //override
        return THIS_C->data();
    }

    INLINE T* data(){
        //override
        return THIS->data();
    }

    template<INT_T... Int>
    INLINE const T* ptr(Int... idx) const{
        return data()+this->offset(idx...);
    }

    template<INT_T... Idx>
    INLINE constexpr const T& operator()(Idx... idx) const noexcept {
        return data()[this->offset(idx...)];
    }

    template<INT_T... Idx>
    INLINE constexpr T& operator()(Idx... idx) noexcept {
        return data()[this->offset(idx...)];
    }

    template<INT_T IDX_T>
    INLINE const T& operator[](IDX_T i) const{
        BOUNDS_ASSERT(i, this->size());
        return data()[i];
    }

    template<typename... Idx>
    INLINE auto operator()(Idx... i) const{
        return tensor_call(THIS_C, i...);
    }

    //MODIFIERS

    template<INT_T IDX_T>
    INLINE T& operator[](IDX_T i){
        BOUNDS_ASSERT(i, this->size());
        return data()[i];
    }

    template<typename... Idx>
    INLINE auto operator()(Idx... i){
        return tensor_call(THIS, i...);
    }

    template<INT_T... Int>
    INLINE T* ptr(Int... idx){
        return data()+this->offset(idx...);
    }

};


template<typename T, Layout L = Layout::C, size_t... DIMS>
class NdView : public AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>{

    using Base = AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit NdView(T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<INT_T... Args>
    explicit NdView(T* data, Args... shape) : Base(shape...), _data(data) {}

    template<IsShapeContainer ShapeContainer>
    explicit NdView(T* data, const ShapeContainer& shape) : Base(shape), _data(data) {}

    INLINE const T* data() const{
        return _data;
    }

    INLINE T* data() {
        return _data;
    }

    void set_data(T* data) {
        _data = data;
    }

private:

    T* _data;
};

template<typename T, size_t Size=0>
using View1D = NdView<T, Layout::C, Size>;

template<typename Derived, INT_T... Int>
inline Derived::value_type& tensor_call(Derived& x, Int... idx){
    return x(idx...);
}

template<typename Derived, INT_T... Int>
inline const Derived::value_type& tensor_call(const Derived& x, Int... idx){
    return x(idx...);
}