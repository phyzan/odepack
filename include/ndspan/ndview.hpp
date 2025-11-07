#pragma once

#include "layoutmap.hpp"

template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractNdView : public NdSpan<T, L, DIMS...>{

    using CLS = AbstractNdView<Derived, L, T, DIMS...>;
    using Base = NdSpan<T, L, DIMS...>;

protected:

    using Base::Base;

public:

    inline const T* data() const{
        //override
        return THIS_C->data();
    }

    inline T* data() {
        return CONST_CAST(T*, data());
    }

    inline const T* begin() const{ return data();}

    inline T* begin() {return data();}

    inline const T* end() const {return data()+size();}

    inline T* end() {return data()+size();}

    template<INT_T... Idx>
    inline constexpr const T& operator()(Idx... idx) const noexcept {
        return data()[this->offset(idx...)];
    }

    template<INT_T... Idx>
    inline constexpr T& operator()(Idx... idx) noexcept {
        return data()[this->offset(idx...)];
    }

    inline const T& operator[](size_t i) const{
        return data()[i];
    }

    inline T& operator[](size_t i){
        return data()[i];
    }


    // function definitions that only assist C++ parsers

    inline constexpr size_t size() const{
        return Base::size();
    }

    inline constexpr size_t ndim() const{
        return Base::ndim();
    }

    inline const size_t* shape() const {
        return Base::shape();
    }

    inline constexpr size_t shape(size_t i) const {
        return Base::shape(i);
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const noexcept {
        return Base::offset(idx...);
    }

    template<INT_T... Args>
    inline void reshape(Args... shape){
        return Base::reshape(shape...);
    }

    template<INT_T... Args>
    inline void resize(Args... shape){
        return Base::resize(shape...);
    }

};


template<typename T, Layout L, size_t... DIMS>
class NdView : public AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>{

    using Base = AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit NdView(T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<INT_T... Args>
    explicit NdView(T* data, Args... shape) : Base(shape...), _data(data) {
        static_assert(sizeof...(shape) > 0, "NdView: at least one dimension (shape) must be provided");
    }

    inline const T* data() const{
        return _data;
    }

    inline T* data() {
        return _data;
    }

    void set_data(T* data) {
        _data = data;
    }

private:

    T* _data;
};