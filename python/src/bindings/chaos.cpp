#include "../../../include/pyode/lib/PyChaos.hpp"

using namespace ode;

PYBIND11_MODULE(chaos, m) {


py::class_<PyVarSolver, PySolver>(m, "VariationalSolver")
    .def(py::init<PyVarSolver>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, std::string, std::string>(),
        py::arg("f"),
        py::arg("jac"),
        py::arg("t0"),
        py::arg("q0"),
        py::arg("period"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("method")="RK45",
        py::arg("scalar_type")="double")
    .def_property_readonly("logksi", &PyVarSolver::py_logksi)
    .def_property_readonly("lyap", &PyVarSolver::py_lyap)
    .def_property_readonly("t_lyap", &PyVarSolver::py_t_lyap)
    .def_property_readonly("delta_s", &PyVarSolver::py_delta_s);


py::class_<PyVarODE, PyODE>(m, "VariationalLowLevelODE")
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::arg("period"),
        py::kw_only(),
        py::arg("jac")=py::none(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("events")=py::tuple(),
        py::arg("method")="RK45",
        py::arg("scalar_type")="double")
    .def(py::init<PyVarODE>(), py::arg("ode"))
    .def_property_readonly("t_lyap", &PyVarODE::py_t_lyap)
    .def_property_readonly("lyap", &PyVarODE::py_lyap)
    .def_property_readonly("kicks", &PyVarODE::py_kicks)
    .def("copy", &PyVarODE::copy);

    
}


