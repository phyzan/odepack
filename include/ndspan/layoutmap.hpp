#pragma once

#include "layouts/standard_layouts.hpp"
#include "layouts/morton.hpp"


/*
All classes in the LayoutMap (e.g. AbstractRowMajorSpan) must have this private method:

template<INT_T... Idx>
inline constexpr size_t _offset_impl(Idx... idx) const noexcept
*/
enum class Layout : std::uint8_t { C, F, Z};

template <Layout L, typename T, size_t... DIMS>
struct LayoutMap;

template <typename T, size_t... DIMS>
struct LayoutMap<Layout::C, T, DIMS...> { using type = RowMajorSpan<T, DIMS...>; };

template <typename T, size_t... DIMS>
struct LayoutMap<Layout::F, T, DIMS...> { using type = ColumnMajorSpan<T, DIMS...>; };

template <typename T, size_t... DIMS>
struct LayoutMap<Layout::Z, T, DIMS...> { using type = ZorderNdSpan<T, DIMS...>; };

template <typename T, Layout L, size_t... DIMS>
using NdSpan = LayoutMap<L, T, DIMS...>::type;