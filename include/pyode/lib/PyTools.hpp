#ifndef PYTOOLS_HPP
#define PYTOOLS_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../odepackDecl.hpp"


namespace ode{

#ifdef MPREAL
static const std::string SCALAR_TYPE[4] = {"float", "double", "long double", "mpreal"};
#else
static const std::string SCALAR_TYPE[3] = {"float", "double", "long double"};
#endif

#define DISPATCH(RETURN_TYPE, ...)                                              \
call_dispatch(this->scalar_type, [&]<typename T>() -> RETURN_TYPE {__VA_ARGS__ });


static const std::map<std::string, int> DTYPE_MAP = {
    {"float", 0},
    {"double", 1},
    {"long double", 2},
#ifdef MPREAL
    {"mpreal", 3}
#endif
};

namespace py = pybind11;


struct PyStruct{

    py::function rhs;
    py::function jac;
    py::function mask;
    py::function event;
    std::vector<py::ssize_t> shape;
    py::tuple py_args = py::make_tuple();
    py::array_t<float> array_f;
    py::array_t<double> array_d;
    py::array_t<long double> array_ld;
    bool is_lowlevel = false;

    template<typename T>
    const py::array_t<T>& get_array() const;

    template<typename T>
    py::array_t<T>& get_array();
};


struct DtypeDispatcher{

    DtypeDispatcher(const std::string& dtype_);

    DtypeDispatcher(int dtype_);

    int scalar_type;
};


struct PyFuncWrapper : DtypeDispatcher {

    void* rhs; //Func<T>
    size_t Nsys;
    std::vector<py::ssize_t> output_shape;
    size_t Nargs;
    size_t output_size;

    PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type);

    template<typename T>
    py::object call_impl(const py::object& t, const py::iterable& py_q, py::args py_args) const;

    py::object call(const py::object& t, const py::iterable& py_q, const py::args& py_args) const;
};


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


template<typename T, typename Container>
Container toCPP_Array(const pybind11::iterable &obj);

std::vector<EventOptions> to_Options(const py::iterable& d);

template<typename T>
StepSequence<T> to_step_sequence(const py::object& t_eval);

py::dict to_PyDict(const EventMap& _map);

template<typename Int>
std::vector<Int> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape);

template<typename Scalar, typename ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<py::ssize_t>& q0_shape = {});

template<typename T>
py::array_t<T> array(T* data, const std::vector<py::ssize_t>& shape);

std::vector<py::ssize_t> shape(const py::object& obj);

template<typename T>
inline T open_capsule(const py::capsule& f){
    void* ptr = f.get_pointer();
    if (ptr == nullptr){
        return nullptr;
    }
    else{
        return reinterpret_cast<T>(ptr);
    }
}

template<typename T>
inline void arrcpy(T* dst, const T* src, size_t size){
    for (size_t i=0; i<size; i++){
        dst[i] = src[i];
    }
}

template<typename T>
inline void arrcpy(T* dst, const void* src, size_t size){
    const T* data = static_cast<const T*>(src);
    arrcpy(dst, data, size);
}

template<typename T>
void py_rhs(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
void py_jac(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
void py_mask(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
T py_event(const T& t, const T* q, const T*, const void* obj);

#ifdef MPREAL
// Specializations for mpfr::mpreal

template<>
inline void py_rhs<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "RHS function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.rhs(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
inline void py_jac<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Jacobian function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.jac(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
inline void py_mask<mpfr::mpreal>(mpfr::mpreal* res, const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Mask function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::iterable pyres = p.mask(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args);
    // Convert py::object back to Array<mpfr::mpreal>
    auto arr = toCPP_Array<mpfr::mpreal, Array1D<mpfr::mpreal, 0>>(pyres);
    arrcpy(res, arr.data(), arr.size());
}

template<>
inline mpfr::mpreal py_event<mpfr::mpreal>(const mpfr::mpreal& t, const mpfr::mpreal* q, const mpfr::mpreal*, const void* obj){
    assert(obj != nullptr && "Event function pointer is null");
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, Array<mpfr::mpreal>(q, p.shape.data(), p.shape.size()), *p.py_args).template cast<mpfr::mpreal>();
}

#endif // MPREAL

bool is_sorted(const py::array_t<double>& arr);


template<typename T>
inline std::string get_scalar_type(){
    if constexpr (std::is_same_v<T, float>){
        return "float";
    }
    else if constexpr (std::is_same_v<T, double>){
        return "double";
    }
    else if constexpr (std::is_same_v<T, long double>){
        return "long double";
    }
#ifdef MPREAL
    else if constexpr (std::is_same_v<T, mpfr::mpreal>){
        return "mpreal";
    }
#endif
    else{
        static_assert(false, "Unsupported scalar type T");
    }
}

} // namespace ode

#endif // PYTOOLS_HPP