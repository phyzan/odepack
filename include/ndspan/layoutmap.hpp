#pragma once

#include "layouts/standard_layouts.hpp"


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

template <Layout L, size_t... DIMS>
using NdSpan = LayoutMap<L, DIMS...>::type;

