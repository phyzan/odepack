#include "../../../include/pyodepack.hpp"

namespace ode{




PyFuncWrapper::PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type) : DtypeDispatcher(scalar_type), rhs(open_capsule<void*>(obj)), Nsys(static_cast<size_t>(Nsys)), output_shape(static_cast<size_t>(output_shape.size())), Nargs(static_cast<size_t>(Nargs)) {
    auto output_shape_c = py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast>(output_shape);
    ndspan::copy_array(this->output_shape.data(), output_shape_c.data(), this->output_shape.size());
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
        Array<T> res(nullptr, output_shape.data(), output_shape.size());
        reinterpret_cast<RhsFunc<T>>(this->rhs)(res.data(), py::cast<T>(t), q.data(), args.data());
        return py::cast(res);
    )
}

//===========================================================================================
//                                      DtypeDispatcher
//===========================================================================================


DtypeDispatcher::DtypeDispatcher(const std::string& dtype_){
    this->scalar_type = DTYPE_MAP.at(dtype_);
}

DtypeDispatcher::DtypeDispatcher(ScalarType dtype_) : scalar_type(dtype_) {}


//===========================================================================================
//                                      Helper Functions
//===========================================================================================

template<>
std::vector<py::ssize_t> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape){
    std::vector<py::ssize_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

std::vector<py::ssize_t> shape(const py::object& obj) {
    py::array arr = py::array::ensure(obj);
    const ssize_t* shape_ptr = arr.shape();  // Pointer to shape data
    auto ndim = static_cast<size_t>(arr.ndim());  // Number of dimensions
    std::vector<py::ssize_t> res(shape_ptr, shape_ptr + ndim);
    return res;
}


std::vector<EventOptions> to_Options(const py::iterable& d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        auto opt = py::cast<EventOptions>(item);
        result.emplace_back(opt);
    }
    result.shrink_to_fit();
    return result;
}




} // namespace ode