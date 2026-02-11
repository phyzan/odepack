#ifndef PYTOOLS_IMPL_HPP
#define PYTOOLS_IMPL_HPP

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../ndspan/arrays.hpp"
#include "../lib/PyTools.hpp"
#include "../../ode/Tools_impl.hpp"
#include  "../pycast/pycast.hpp"




namespace ode{


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


//===========================================================================================
//                                      PyFuncWrapper
//===========================================================================================


PyFuncWrapper::PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type) : DtypeDispatcher(scalar_type), rhs(open_capsule<void*>(obj)), Nsys(static_cast<size_t>(Nsys)), output_shape(static_cast<size_t>(output_shape.size())), Nargs(static_cast<size_t>(Nargs)) {
    copy_array(this->output_shape.data(), output_shape.data(), this->output_shape.size());
    long s = 1;
    for (long i : this->output_shape){
        s *= i;
    }
    this->output_size = size_t(s);
}

py::object PyFuncWrapper::call(const py::object& t, const py::iterable& py_q, const py::args& py_args) const{

    return DISPATCH(py::object,
        auto q = toCPP_Array<T, Array1D<T>>(py_q);
        if (static_cast<size_t>(q.size()) != Nsys || py_args.size() != Nargs){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        auto args = toCPP_Array<T, std::vector<T>>(py_args);
        Array<T> res(output_shape.data(), output_shape.size());
        reinterpret_cast<Func<T>>(this->rhs)(res.data(), py::cast<T>(t), q.data(), args.data(), nullptr);
        return py::cast(res);
    )
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


} // namespace ode

#endif