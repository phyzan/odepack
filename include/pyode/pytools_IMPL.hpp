#ifndef PYTOOLS_IMPL_HPP
#define PYTOOLS_IMPL_HPP

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../ndspan/arrays.hpp"
#include "pytools.hpp"
#include  "ode_caster.hpp"




namespace ode{

#define DISPATCH(RETURN_TYPE, ...)                                              \
call_dispatch(this->scalar_type, [&]<typename T>() -> RETURN_TYPE {__VA_ARGS__ });


template<typename Callable>
auto call_dispatch(int scalar_type, Callable&& f){
    switch (scalar_type) {
        case 0:
            return f.template operator()<float>();
        case 1:
            return f.template operator()<double>();
#ifdef MPREAL
        case 2:
            return f.template operator()<long double>();
        default:
            assert(scalar_type == 3 && "Invalid scalar type");
            return f.template operator()<mpfr::mpreal>();
#else
        default:
            assert(scalar_type == 2 && "Invalid scalar type");
            return f.template operator()<long double>();
#endif
    }
}


namespace py = pybind11;

template<typename T, typename Container>
Container toCPP_Array(const pybind11::iterable &obj) {

    // Get Python iterable length if possible (optional, for efficiency)
    size_t len = py::len(obj);
    Container res(len);
    int i = 0;
    for (auto item : obj) {
        // Cast Python object to double safely
        auto val = py::cast<T>(item);
        // Construct T from double
        res[i] = val;
        i++;
    }

    return res;
}

 std::vector<EventOptions> to_Options(const py::iterable& d);


template<typename T>
StepSequence<T> to_step_sequence(const py::object& t_eval){
    if (t_eval.is_none()){
        return StepSequence<T>();
    }
    else if (py::iterable(t_eval)){
        std::vector<T> vec = toCPP_Array<T, std::vector<T>>(t_eval);
        return StepSequence<T>(vec.data(), vec.size());
    }
    else{
        throw py::type_error("Expected None, a NumPy array, or an iterable of numeric values.");
    }
}

 py::dict to_PyDict(const EventMap& _map){
    py::dict py_dict;
    for (const auto& [key, vec] : _map) {
        py::array_t<size_t> np_array(static_cast<py::ssize_t>(vec.size()), vec.data()); // Create NumPy array
        py_dict[key.c_str()] = np_array; // Assign to dictionary
    }
    return py_dict;
}

template<typename Int>
 std::vector<Int> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape){
    std::vector<Int> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(Int(dim1));        // Add the first element
    for (size_t i=0; i<shape.size(); i++){
        result.push_back(shape[i]);
    }
    return result;
}

template<>
 std::vector<py::ssize_t> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape){
    std::vector<py::ssize_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

template<typename Scalar, typename ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<py::ssize_t>& q0_shape){
    if (q0_shape.size() == 0){
        py::array_t<Scalar> res(std::vector<py::ssize_t>{static_cast<py::ssize_t>(array.size())}, array.data());
        return res;
    }
    else{
        py::array_t<Scalar> res(q0_shape, array.data());
        return res;
    }
}

template<typename T>
py::array_t<T> array(T* data, const std::vector<py::ssize_t>& shape){
    py::capsule capsule = py::capsule(data, [](void* r){T* d = reinterpret_cast<T*>(r); delete[] d;});
    return py::array_t<T>(shape, data, capsule);
}


 std::vector<py::ssize_t> shape(const py::object& obj) {
    py::array arr = py::array::ensure(obj);
    const ssize_t* shape_ptr = arr.shape();  // Pointer to shape data
    auto ndim = static_cast<size_t>(arr.ndim());  // Number of dimensions
    std::vector<py::ssize_t> res(shape_ptr, shape_ptr + ndim);
    return res;
}

template<typename T>
T open_capsule(const py::capsule& f){
    void* ptr = f.get_pointer();
    if (ptr == nullptr){
        return nullptr;
    }
    else{
        return reinterpret_cast<T>(ptr);
    }
}


template<typename T>
const py::array_t<T>& PyStruct::get_array() const {
    if constexpr (std::is_same_v<T, float>) {
        return array_f;
    } else if constexpr (std::is_same_v<T, double>) {
        return array_d;
    } else if constexpr (std::is_same_v<T, long double>) {
        return array_ld;
    } else {
        static_assert(false, "Unsupported type for PyStruct array.");
        return array_d; // This line will never be reached
    }
}

template<typename T>
py::array_t<T>& PyStruct::get_array() {
    return const_cast<py::array_t<T>&>(static_cast<const PyStruct&>(*this).get_array<T>());
}


template<typename T>
void arrcpy(T* dst, const T* src, size_t size){
    for (size_t i=0; i<size; i++){
        dst[i] = src[i];
    }
}

template<typename T>
void arrcpy(T* dst, const void* src, size_t size){
    const T* data = static_cast<const T*>(src);
    arrcpy(dst, data, size);
}

template<typename T>
void py_rhs(T* res, const T& t, const T* q, const T*, const void* obj){
    assert(obj != nullptr && "RHS function pointer is null");
    PyStruct& p = *const_cast<PyStruct*>(reinterpret_cast<const PyStruct*>(obj));
    py::array_t<T>& arr = p.get_array<T>();
    std::memcpy(arr.mutable_data(), q, arr.size() * sizeof(T));
    py::array_t<T> pyres = p.rhs(t, arr, *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_jac(T* res, const T& t, const T* q, const T*, const void* obj){
    assert(obj != nullptr && "Jacobian function pointer is null");
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.jac(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_mask(T* res, const T& t, const T* q, const T*, const void* obj){
    //args should always be the same as p.py_args
    assert(obj != nullptr && "Mask function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.mask(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
T py_event(const T& t, const T* q, const T*, const void* obj){
    //args should always be the same as p.py_args
    assert(obj != nullptr && "Event function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, py::array_t<T>(p.shape, q), *p.py_args).template cast<T>();
}


#ifdef MPREAL
// Specializations for mpfr::mpreal

template<>
void py_rhs<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "RHS function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.rhs(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
void py_jac<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Jacobian function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.jac(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
void py_mask<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Mask function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.mask(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
mpfr::mpreal py_event<mpfr::mpreal>(const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Event function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args).template cast<mpfr::mpreal>();
}

#endif // MPREAL

} // namespace ode

#endif