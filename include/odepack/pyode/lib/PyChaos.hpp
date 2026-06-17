#ifndef PY_CHAOS_HPP
#define PY_CHAOS_HPP

#include "PyOde.hpp"
#include "PySolver.hpp"
#include "../../ode/Chaos/VariationalSolvers.hpp"

namespace ode::python {

struct PyVarSolver : public PySolver{

    PyVarSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::iterable& py_delta_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const std::string& method, const std::string& scalar_type);

    PyVarSolver(void* solver, PyStruct py_data, ScalarType scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarSolver)

    py::object py_logksi() const;

    py::object py_lyap() const;

    py::object py_t_lyap() const;

    py::object py_delta_s() const;

    py::object copy() const override;

    template<typename T>
    chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* cast();

    template<typename T>
    const chaos::ChaoticSolver<T, 0, SolverPolicy::RichVirtual>* cast() const;

};

class PyVarODE : public PyODE{

public:

    PyVarODE(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& q0, const py::iterable& delta_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    template<typename T>
    chaos::VariationalODE<T, 0>& varode();

    template<typename T>
    const chaos::VariationalODE<T, 0>& varode() const;

    py::object py_t_lyap() const;

    py::object py_lyap() const;

    py::object py_kicks() const;

    py::object copy() const override;
    
};


} // namespace ode::python

#endif