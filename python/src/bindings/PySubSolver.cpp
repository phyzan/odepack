#include "odetemplates.hpp"



namespace ode{

//===========================================================================================
//                                      PyRK23
//===========================================================================================

PyRK23::PyRK23(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

PyRK23::PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK23", scalar_type){
}

py::object PyRK23::copy() const{
    return py::cast(PyRK23(*this));
}

//===========================================================================================
//                                      PyRK45
//===========================================================================================

PyRK45::PyRK45(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

PyRK45::PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK45", scalar_type){}

py::object PyRK45::copy() const{
    return py::cast(PyRK45(*this));
}

//===========================================================================================
//                                      PyDOP853
//===========================================================================================

PyDOP853::PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "DOP853", scalar_type){}

PyDOP853::PyDOP853(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

py::object PyDOP853::copy() const{
    return py::cast(PyDOP853(*this));
}

//===========================================================================================
//                                      PyBDF
//===========================================================================================

PyBDF::PyBDF(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(f, jac, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "BDF", scalar_type){}

PyBDF::PyBDF(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

py::object PyBDF::copy() const{
    return py::cast(PyBDF(*this));
}

//===========================================================================================
//                                      PyRK4
//===========================================================================================

PyRK4::PyRK4(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK4", scalar_type){}

PyRK4::PyRK4(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

py::object PyRK4::copy() const{
    return py::cast(PyRK4(*this));
}

} // namespace ode