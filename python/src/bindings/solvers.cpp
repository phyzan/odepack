#include "../../../include/pyode/lib/PySubSolver.hpp"
#include "../../../include/pyode/lib/PyTools.hpp"

using namespace ode;

PYBIND11_MODULE(solvers, m) {

py::class_<PyFuncWrapper>(m, "LowLevelFunction")
    .def(py::init<py::capsule, py::ssize_t, py::array_t<py::ssize_t>, py::ssize_t, py::str>(),
        py::arg("pointer"),
        py::arg("input_size"),
        py::arg("output_shape"),
        py::arg("Nargs"),
        py::arg("scalar_type")="double")
    .def("__call__", &PyFuncWrapper::call, py::arg("t"), py::arg("q"))
    .def_property_readonly("scalar_type", [](const PyFuncWrapper& self){return getScalarType(self.scalar_type);});

py::class_<PyConstSolver>(m, "OdeSolverView")
    .def_property_readonly("t0", &PyConstSolver::t0)
    .def_property_readonly("q0", &PyConstSolver::q0)
    .def_property_readonly("direction", &PyConstSolver::direction)
    .def_property_readonly("t", &PyConstSolver::t)
    .def_property_readonly("q", &PyConstSolver::q)
    .def_property_readonly("t_last", &PyConstSolver::t_last)
    .def_property_readonly("t_old", &PyConstSolver::t_old)
    .def_property_readonly("q_last", &PyConstSolver::q_last)
    .def_property_readonly("q_old", &PyConstSolver::q_old)
    .def_property_readonly("stepsize", &PyConstSolver::stepsize)
    .def_property_readonly("diverges", &PyConstSolver::diverges)
    .def_property_readonly("is_dead", &PyConstSolver::is_dead)
    .def_property_readonly("Nsys", &PyConstSolver::Nsys)
    .def_property_readonly("n_evals_rhs", &PyConstSolver::n_evals_rhs)
    .def_property_readonly("n_evals_jac", &PyConstSolver::n_evals_jac)
    .def_property_readonly("status", &PyConstSolver::status)
    .def("at_event", &PyConstSolver::py_at_event, py::arg("event")=py::none())
    .def("show_state", &PyConstSolver::show_state,
        py::arg("digits") = 8
    )
    .def("rhs", &PyConstSolver::py_rhs, py::arg("t"), py::arg("q"))
    .def("jac", &PyConstSolver::py_jac, py::arg("t"), py::arg("q"))
    .def("timeit_rhs", &PyConstSolver::timeit_rhs, py::arg("t"), py::arg("q"))
    .def("timeit_jac", &PyConstSolver::timeit_jac, py::arg("t"), py::arg("q"))
    .def("copy", &PyConstSolver::copy)
    .def_property_readonly("scalar_type", [](const PyConstSolver& self){return getScalarType(self.scalar_type);});

py::class_<PySolver, PyConstSolver>(m, "OdeSolver")
    .def(py::init<PyConstSolver>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string, std::string>(),
        py::arg("f"),
        py::arg("jac")=py::none(),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
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
    .def("advance", &PySolver::advance)
    .def("timeit_step", &PySolver::timeit_step)
    .def("advance_to_event", &PySolver::advance_to_event, py::arg("event")=py::none())
    .def("advance_until", &PySolver::advance_until, py::arg("t"), py::arg("observer")=py::none(), py::arg("extra_steps")=py::none())
    .def("reset", &PySolver::reset)
    .def("set_ics", &PySolver::set_ics, py::arg("t0"), py::arg("q0"), py::arg("stepsize")=0, py::arg("direction")=0)
    .def("resume", &PySolver::resume)
    .def("stop", &PySolver::stop, py::arg("reason"))
    .def("kill", &PySolver::kill, py::arg("reason"));


py::class_<PyRK23, PySolver>(m, "RK23")
    .def(py::init<PyRK23>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("events")=py::tuple(),
        py::arg("scalar_type")="double");

py::class_<PyRK45, PySolver>(m, "RK45")
    .def(py::init<PyRK45>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("events")=py::tuple(),
        py::arg("scalar_type")="double");

py::class_<PyDOP853, PySolver>(m, "DOP853")
    .def(py::init<PyDOP853>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("events")=py::tuple(),
        py::arg("scalar_type")="double");

py::class_<PyBDF, PySolver>(m, "BDF")
    .def(py::init<PyBDF>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
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
        py::arg("scalar_type")="double");


py::class_<PyRK4, PySolver>(m, "RK4")
    .def(py::init<PyRK4>(), py::arg("solver"))
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("args")=py::tuple(),
        py::arg("events")=py::tuple(),
        py::arg("scalar_type")="double");

m.def("advance_all", &py_advance_all, py::arg("solvers"), py::arg("t_goal"), py::arg("threads")=-1, py::arg("display_progress")=false);

m.def("advance_all_to_event", &py_advance_all_to_event, py::arg("solvers"), py::arg("event"), py::arg("tmax"), py::arg("threads")=-1, py::arg("display_progress")=false);


#ifdef MPREAL
    m.def("set_mpreal_prec",
      &mpfr::mpreal::set_default_prec,
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", &mpfr::mpreal::get_default_prec);
#else
    m.def("set_mpreal_prec",
      [](size_t){
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      });
#endif

} // PYBIND11_MODULE(solvers, m)
