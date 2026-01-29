#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef MPREAL
#include <mpreal.h>
#endif
#include <vector>
#include <type_traits>

namespace py = pybind11;

// namespace pybind11::detail {

// // ====================== mpreal <-> mpmath.mpf caster ======================
// template <>
// struct type_caster<mpfr::mpreal> {
// public:
//     PYBIND11_TYPE_CASTER(mpfr::mpreal, _("mpmath.mpf"));

//     // Python -> C++
//     bool load(handle src, bool) {
//         try {
//             static py::object mpmath = py::module_::import("mpmath");
//             static py::object mpf = mpmath.attr("mpf");

//             py::object val = mpf(src);
//             std::string s = py::str(val);
//             value = mpfr::mpreal(s);
//             return true;
//         } catch (...) {
//             return false;
//         }
//     }

//     // C++ -> Python
//     static handle cast(const mpfr::mpreal &src,
//                        return_value_policy,
//                        handle) {
//         static py::object mpmath = py::module_::import("mpmath");
//         static py::object mpf = mpmath.attr("mpf");

//         std::string s = src.toString(); // full precision
//         py::object result = mpf(py::str(s));
//         return result.release();
//     }
// };

// // ====================== ArrayType<T> caster ======================
// // Assumes ArrayType<T> has:
// //   - .data() -> const T*
// //   - .shape() -> const size_t*
// //   - .ndim() -> size_t
// //   - shape(i) -> size_t

// // Check if type has the required array interface methods
// template <typename T, typename = void>
// struct has_array_interface : std::false_type {};

// template <typename T>
// struct has_array_interface<T, std::void_t<
//     typename T::value_type,
//     decltype(std::declval<const T>().data()),
//     decltype(std::declval<const T>().ndim()),
//     decltype(std::declval<const T>().shape(size_t(0)))
// >> : std::true_type {};

// template <typename ArrayType>
// struct type_caster<ArrayType, std::enable_if_t<
//     has_array_interface<ArrayType>::value &&
//     !std::is_base_of_v<py::array_t<typename ArrayType::value_type>, ArrayType>>> {

//     PYBIND11_TYPE_CASTER(ArrayType, _("ArrayType"));

//     bool load(handle /*src*/, bool /*convert*/) {
//         // Optional: implement Python -> ArrayType<T> if desired
//         return false;
//     }

//     static handle cast(const ArrayType &src,
//                        return_value_policy /* policy */,
//                        handle /* parent */)
//     {
//         using T = typename ArrayType::value_type;
//         constexpr bool is_mpreal = std::is_same_v<T, mpfr::mpreal>;
//         const size_t ndim = src.ndim();
//         // Build shape vector
//         std::vector<ssize_t> shape(ndim);
//         for (size_t i = 0; i < ndim; ++i) {
//             shape[i] = static_cast<ssize_t>(src.shape(i));
//         }

//         if constexpr (is_mpreal) {
//             // For mpreal: numpy array of Python objects
//             size_t total_size = 1;
//             for (auto s : shape) { total_size *= size_t(s); }

//             // Create numpy array with object scalar_type
//             py::array result = py::array(py::scalar_type("O"), shape);

//             // Only populate if size > 0

//             if (src.size() > 0) {
//                 const mpfr::mpreal* data = src.data();
//                 auto* out_ptr = static_cast<py::object*>(result.mutable_data());

//                 try {
//                     py::object mpmath = py::module_::import("mpmath");
//                     py::object mpf = mpmath.attr("mpf");
//                     for (size_t i = 0; i < total_size; ++i) {
//                         out_ptr[i] = mpf(py::str(data[i].toString()));
//                     }
//                 } catch (...) {
//                     // Fallback: use strings if mpmath is unavailable
//                     for (size_t i = 0; i < total_size; ++i) {
//                         out_ptr[i] = py::str(data[i].toString());
//                     }
//                 }
//             }

//             return result.release();

//         } else if constexpr (std::is_same_v<std::remove_cv_t<T>, mpfr::mpreal>) {
//             // For const mpreal: return object array
//             size_t total_size = 1;
//             for (auto s : shape) { total_size *= size_t(s); }

//             py::array result = py::array(py::scalar_type("O"), shape);

//             // Only populate if size > 0
//             if (src.size() > 0) {
//                 auto* out_ptr = static_cast<py::object*>(result.mutable_data());

//                 try {
//                     py::object mpmath = py::module_::import("mpmath");
//                     py::object mpf = mpmath.attr("mpf");
//                     for (size_t i = 0; i < total_size; ++i) {
//                         out_ptr[i] = mpf(py::str(src.data()[i].toString()));
//                     }
//                 } catch (...) {
//                     // Fallback: use strings if mpmath is unavailable
//                     for (size_t i = 0; i < total_size; ++i) {
//                         out_ptr[i] = py::str(src.data()[i].toString());
//                     }
//                 }
//             }

//             return result.release();
//         } else {
//             // For POD types like double/int: numpy array of T
//             return py::array_t<T>(shape, src.data()).release();
//         }
//     }
// };

// } // namespace pybind11::detail

namespace pybind11::detail {

#ifdef MPREAL
// ====================== mpreal <-> double caster ======================
template <>
struct type_caster<mpfr::mpreal> {
public:
    PYBIND11_TYPE_CASTER(mpfr::mpreal, _("float"));

    // Python -> C++
    bool load(handle src, bool) {
        try {
            double val = py::cast<double>(src);
            value = mpfr::mpreal(val);
            return true;
        } catch (...) {
            return false;
        }
    }

    // C++ -> Python
    static handle cast(const mpfr::mpreal &src,
                       return_value_policy,
                       handle) {
        double val = src.toDouble();
        return py::cast(val).release();
    }
};
#endif // MPREAL

// ====================== ArrayType<T> caster ======================
// Assumes ArrayType<T> has:
//   - .data() -> const T*
//   - .shape() -> const size_t*
//   - .ndim() -> size_t
//   - shape(i) -> size_t

// Check if type has the required array interface methods
template <typename T, typename = void>
struct has_array_interface : std::false_type {};

template <typename T>
struct has_array_interface<T, std::void_t<
    typename T::value_type,
    decltype(std::declval<const T>().data()),
    decltype(std::declval<const T>().ndim()),
    decltype(std::declval<const T>().shape(size_t(0)))
>> : std::true_type {};

template <typename ArrayType>
struct type_caster<ArrayType, std::enable_if_t<
    has_array_interface<ArrayType>::value &&
    !std::is_base_of_v<py::array_t<typename ArrayType::value_type>, ArrayType>>> {

    PYBIND11_TYPE_CASTER(ArrayType, _("ArrayType"));

    bool load(handle /*src*/, bool /*convert*/) {
        // Optional: implement Python -> ArrayType<T> if desired
        return false;
    }

    static handle cast(const ArrayType &src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        using T = typename ArrayType::value_type;

#ifdef MPREAL
        constexpr bool is_mpreal = std::is_same_v<T, mpfr::mpreal>;
#endif
        const size_t ndim = src.ndim();
        // Build shape vector
        std::vector<ssize_t> shape(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape[i] = static_cast<ssize_t>(src.shape(i));
        }
#ifdef MPREAL
        if constexpr (is_mpreal) {
            // For mpreal: numpy array of doubles
            size_t total_size = 1;
            for (auto s : shape) { total_size *= size_t(s); }

            // Create numpy array with double scalar_type
            py::array_t<double> result(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);

            if (src.size() > 0) {
                const mpfr::mpreal* data = src.data();
                for (size_t i = 0; i < total_size; ++i) {
                    ptr[i] = data[i].toDouble();
                }
            }

            return result.release();

        } else if constexpr (std::is_same_v<std::remove_cv_t<T>, mpfr::mpreal>) {
            // For const mpreal: return double array
            size_t total_size = 1;
            for (auto s : shape) { total_size *= size_t(s); }

            py::array_t<double> result(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);

            if (src.size() > 0) {
                const mpfr::mpreal* data = src.data();
                for (size_t i = 0; i < total_size; ++i) {
                    ptr[i] = data[i].toDouble();
                }
            }

            return result.release();
        }else{
            return py::array_t<T>(shape, src.data()).release();
        }
#else
        // Default: For POD types like double/int: numpy array of T
        return py::array_t<T>(shape, src.data()).release();
#endif
    }
};

} // namespace pybind11::detail
