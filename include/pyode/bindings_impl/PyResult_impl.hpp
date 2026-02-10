#ifndef PY_RESULT_IMPL_HPP
#define PY_RESULT_IMPL_HPP

#include "../bindings/PyResult.hpp"
#include "../pytools/pycast.hpp"

namespace ode{


// ============================================================================================
//                                      PyOdeResult
// ============================================================================================


template<typename T>
const OdeResult<T>* PyOdeResult::cast() const{
    return reinterpret_cast<OdeResult<T>*>(this->res);
}

template<typename T>
OdeResult<T>* PyOdeResult::cast() {
    return reinterpret_cast<OdeResult<T>*>(this->res);
}

// ============================================================================================
//                                      PyOdeSolution
// ============================================================================================


template<typename T>
py::object PyOdeSolution::_get_frame(const py::object& t) const{
    return py::cast(Array<T>(reinterpret_cast<OdeSolution<T>*>(this->res)->operator()(t.cast<T>()).data(), this->q0_shape.data(), this->q0_shape.size()));
}

template<typename T>
py::object PyOdeSolution::_get_array(const py::array& py_array) const{
    const auto nt = size_t(py_array.size());
    std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
    final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
    Array<T> res(final_shape.data(), final_shape.size());
    const auto* solution = reinterpret_cast<const OdeSolution<T>*>(this->res);

    // Extract array values and cast them to T using Python's item access
    for (size_t i=0; i<nt; i++){
        py::object item = py_array.attr("flat")[py::int_(i)];
        T t_value = py::cast<T>(item);
        copy_array(res.data()+i*nsys, solution->operator()(t_value).data(), nsys);
    }
    return py::cast(res);
}

} // namespace ode

#endif // PY_RESULT_IMPL_HPP