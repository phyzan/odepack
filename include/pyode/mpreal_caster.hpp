#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpreal.h>

namespace pybind11::detail {

template <>
struct type_caster<mpfr::mpreal> {
public:
    PYBIND11_TYPE_CASTER(mpfr::mpreal, _("mpreal"));

    // Python → C++
    bool load(handle src, bool) {
        // Accept Python float or int
        if (!src){ return false;}
        try {
            value = mpfr::mpreal(pybind11::cast<double>(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    // C++ → Python
    static handle cast(const mpfr::mpreal &src,
                       return_value_policy,
                       handle) {
        return pybind11::float_(static_cast<double>(src)).release();
    }
};

// Numpy array support for mpfr::mpreal
// We'll use double as the numpy dtype since mpreal is not a POD type
template <>
struct npy_format_descriptor<mpfr::mpreal> {
    static constexpr auto name = _("float64");
    static pybind11::dtype dtype() {
        return pybind11::dtype::of<double>();
    }
};

} // namespace pybind11::detail
