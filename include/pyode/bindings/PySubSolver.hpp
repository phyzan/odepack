#ifndef PY_SUBSOLVER_HPP
#define PY_SUBSOLVER_HPP


#include "PySolver.hpp"

namespace ode {


struct PyRK23 : public PySolver{

    PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK23(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK23)

    py::object copy() const override;

};


struct PyRK45 : public PySolver{

    PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK45(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK45)

    py::object copy() const override;

};


struct PyDOP853 : public PySolver{

    PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyDOP853(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyDOP853)

    py::object copy() const override;

};


struct PyBDF : public PySolver{

    PyBDF(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyBDF(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyBDF)

    py::object copy() const override;

};

struct PyRK4 : public PySolver{

    PyRK4(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK4(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK4)

    py::object copy() const override;

};

} // namespace ode


#endif // PY_SUBSOLVER_HPP