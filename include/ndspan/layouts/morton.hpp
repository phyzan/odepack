#pragma once

#include "../ndspan.hpp"

template<typename T, size_t... DIMS>
class ZorderNdSpan : public DerivedNdSpan<ZorderNdSpan<T, DIMS...>, T, DIMS...>{

};