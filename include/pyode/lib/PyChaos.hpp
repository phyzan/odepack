#ifndef PY_CHAOS_HPP
#define PY_CHAOS_HPP

#include "PyOde.hpp"
#include "PySolver.hpp"
#include "../../ode/Chaos/VariationalSolvers.hpp"

namespace ode{

struct PyVarSolver : public PySolver{

    PyVarSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const std::string& method, const std::string& scalar_type);

    PyVarSolver(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarSolver)

    template<typename T>
    const NormalizationEvent<T>& main_event() const;

    py::object py_logksi() const;

    py::object py_lyap() const;

    py::object py_t_lyap() const;

    py::object py_delta_s() const;

    py::object copy() const override;

};

class PyVarODE : public PyODE{

public:

    PyVarODE(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& period, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    template<typename T>
    VariationalODE<T, 0, Func<T>, Func<T>>& varode();

    template<typename T>
    const VariationalODE<T, 0, Func<T>, Func<T>>& varode() const;

    py::object py_t_lyap() const;

    py::object py_lyap() const;

    py::object py_kicks() const;

    py::object copy() const override;
    
};


} // namespace ode

#endif