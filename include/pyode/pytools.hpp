#ifndef PYTOOLS_HPP
#define PYTOOLS_HPP

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "include/ndspan/arrays.hpp"

namespace py = pybind11;


inline std::vector<py::ssize_t> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape){
    std::vector<py::ssize_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

template<typename Scalar, typename ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<py::ssize_t>& q0_shape = {}){
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

template<typename T, typename Container>
Container toCPP_Array(const pybind11::iterable &obj) {

    // Get Python iterable length if possible (optional, for efficiency)
    size_t len = pybind11::len(obj);
    Container res(len);
    int i = 0;
    for (auto item : obj) {
        // Cast Python object to double safely
        double val = pybind11::cast<double>(item);
        // Construct T from double
        res[i] = val;
        i++;
    }

    return res;
}


inline std::vector<py::ssize_t> shape(const py::object& obj) {
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


struct PyStruct{

    py::function rhs;
    py::function jac;
    py::function mask;
    py::function event;
    std::vector<py::ssize_t> shape;
    py::tuple py_args = py::make_tuple();

    bool operator==(const PyStruct& other) const {
        std::vector<py::function> _rhs = {this->rhs, jac, mask, event};
        std::vector<py::function> _lhs = {other.rhs, other.jac, other.mask, other.event};
        for (size_t i=0; i<_rhs.size(); i++){
            if (!(PyObject_RichCompareBool(_rhs[i].ptr(), _lhs[i].ptr(), Py_EQ) == 1)){
                return false;
            }
        }

        return (PyObject_RichCompareBool(py_args.ptr(), other.py_args.ptr(), Py_EQ) == 1) && (shape==other.shape);
    }

};


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
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.rhs(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_jac(T* res, const T& t, const T* q, const T*, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.jac(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_mask(T* res, const T& t, const T* q, const T*, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.mask(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
T py_event(const T& t, const T* q, const T*, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, py::array_t<T>(p.shape, q), *p.py_args).template cast<T>();
}

// Specializations for mpfr::mpreal

template<>
void py_rhs<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.rhs(t, Array<mpfr::mpreal>(q, p.shape), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
void py_jac<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.rhs(t, Array<mpfr::mpreal>(q, p.shape), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
void py_mask<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.rhs(t, Array<mpfr::mpreal>(q, p.shape), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
mpfr::mpreal py_event<mpfr::mpreal>(const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, Array<mpfr::mpreal>(q, p.shape), *p.py_args).template cast<mpfr::mpreal>();
}

#endif