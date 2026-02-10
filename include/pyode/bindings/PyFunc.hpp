#ifndef PY_FUNC_HPP
#define PY_FUNC_HPP

#include "../pytools/pytools.hpp"

namespace ode{

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

} // namespace ode


#endif // PY_FUNC_HPP