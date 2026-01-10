#pragma once

#include "layoutmap.hpp"



template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractView : public NdSpan<L, DIMS...>{

    using CLS = AbstractView<Derived, L, T, DIMS...>;
    using Base = NdSpan<L, DIMS...>;

protected:

    using Base::Base;

    // place resize in protected, so that derived classes resize their internal data before resizing in NdSpan
    template<INT_T... Args>
    INLINE void constexpr resize(Args... shape){
        Base::resize(shape...);
    }

    template<INT_T Int>
    INLINE void resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim);
    }

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    //ACCESSORS

    const_iterator begin() const { return this->data(); }
    const_iterator end() const { return this->data() + this->size(); }
    
    INLINE const T* data() const{
        //override
        return THIS_C->data();
    }

    template<INT_T... Int>
    INLINE const T* ptr(Int... idx) const{
        return data()+this->offset(idx...);
    }

    template<INT_T... Idx>
    INLINE constexpr const T& operator()(Idx... idx) const {
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

};

template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractMutView : public AbstractView<Derived, L, T, DIMS...>{

    using CLS = AbstractMutView<Derived, L, T, DIMS...>;
    using Base = AbstractView<Derived, L, T, DIMS...>;

protected:

    using Base::Base;

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    //MODIFIERS

    iterator begin() { return this->data(); }
    iterator end()   { return this->data() + this->size(); }

    INLINE T* data(){
        //override
        return THIS->data();
    }

    template<INT_T IDX_T>
    INLINE T& operator[](IDX_T i){
        BOUNDS_ASSERT(i, this->size());
        return data()[i];
    }

    template<INT_T... Int>
    INLINE T* ptr(Int... idx){
        return data()+this->offset(idx...);
    }

    template<INT_T... Idx>
    INLINE constexpr T& operator()(Idx... idx) noexcept {
        return data()[this->offset(idx...)];
    }

    template<typename... Idx>
    INLINE auto operator()(Idx... i){
        return tensor_call(THIS, i...);
    }

    using Base::data;
    using Base::operator();
    using Base::operator[];
    using Base::ptr;

};


template<typename T, Layout L = Layout::C, size_t... DIMS>
class View : public AbstractView<View<T, L, DIMS...>, L, T, DIMS...>{

    using Base = AbstractView<View<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit View(const T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<INT_T... Args>
    explicit View(const T* data, Args... shape) : Base(shape...), _data(data) {}

    template<INT_T Int>
    explicit View(const T* data, const Int* shape, size_t ndim) : Base(shape, ndim), _data(data) {}


    INLINE const T* data() const{
        return _data;
    }

    void set_data(const T* data) {
        _data = data;
    }

private:

    const T* _data;
};


template<typename T, Layout L = Layout::C, size_t... DIMS>
class MutView : public AbstractMutView<MutView<T, L, DIMS...>, L, T, DIMS...>{

    using Base = AbstractMutView<MutView<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit MutView(T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<INT_T... Args>
    explicit MutView(T* data, Args... shape) : Base(shape...), _data(data) {}

    template<INT_T Int>
    explicit MutView(T* data, const Int* shape, size_t ndim) : Base(shape, ndim), _data(data) {}

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
using MutView1D = MutView<T, Layout::C, Size>;

template<typename T, size_t Size=0>
using View1D = View<T, Layout::C, Size>;

template<typename Derived, INT_T... Int>
inline Derived::value_type& tensor_call(Derived& x, Int... idx){
    return x(idx...);
}

template<typename Derived, INT_T... Int>
inline const Derived::value_type& tensor_call(const Derived& x, Int... idx){
    return x(idx...);
}