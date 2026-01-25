#pragma once

#include "layouts/standard_layouts.hpp"
#include "layouts/morton.hpp"


namespace ndspan{
/*
All classes in the LayoutMap (e.g. AbstractRowMajorSpan) must have this private method:

template<INT_T... Idx>
INLINE constexpr size_t _offset_impl(Idx... idx) const noexcept
*/
enum class Layout : std::uint8_t { C, F, Z};

template <Layout L, size_t... DIMS>
struct LayoutMap;

template <size_t... DIMS>
struct LayoutMap<Layout::C, DIMS...> { using type = RowMajorSpan<DIMS...>; };

template <size_t... DIMS>
struct LayoutMap<Layout::F, DIMS...> { using type = ColumnMajorSpan<DIMS...>; };

template <size_t... DIMS>
struct LayoutMap<Layout::Z, DIMS...> { using type = ZorderNdSpan<DIMS...>; };

template <Layout L, size_t... DIMS>
class NdSpan : public std::conditional_t<(sizeof...(DIMS)==1 && (DIMS*...*1)==0), SemiStaticSpan1D, typename LayoutMap<L, DIMS...>::type>{

    using Base = std::conditional_t<(sizeof...(DIMS)==1 && (DIMS*...*1)==0), SemiStaticSpan1D, typename LayoutMap<L, DIMS...>::type>;

protected:

    inline static constexpr size_t ND = Base::ND;
    inline static constexpr size_t N = Base::N;
    inline static constexpr std::array<size_t, ND> SHAPE = Base::SHAPE;

public:

    template<INT_T Int>
    explicit NdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<INT_T... Args>
    explicit constexpr NdSpan(Args... shape) : Base(shape...){}

    //ACCESSORS
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

    //MODIFIERS
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
};


} // namespace ndspan