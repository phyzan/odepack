#include "odetemplates.hpp"

namespace ode{

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

} // namespace ode