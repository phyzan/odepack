#ifndef PYTOOLS_HPP
#define PYTOOLS_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../odepack_.hpp"


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
T open_capsule(const py::capsule& f);

template<typename T>
void arrcpy(T* dst, const T* src, size_t size);

template<typename T>
void arrcpy(T* dst, const void* src, size_t size);

template<typename T>
void py_rhs(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
void py_jac(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
void py_mask(T* res, const T& t, const T* q, const T*, const void* obj);

template<typename T>
T py_event(const T& t, const T* q, const T*, const void* obj);

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