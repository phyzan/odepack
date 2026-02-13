#include "../../../include/pyode/lib/PyOde.hpp"


using namespace ode;

PYBIND11_MODULE(ode, m) {


py::class_<PyODE>(m, "LowLevelODE")
    .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
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
        py::arg("method")="RK45",
        py::arg("scalar_type")="double")
    .def(py::init<PyODE>(), py::arg("ode"))
    .def("rhs", &PyODE::call_Rhs, py::arg("t"), py::arg("q"))
    .def("jac", &PyODE::call_Jac, py::arg("t"), py::arg("q"))
    .def("solver", &PyODE::solver_copy, py::keep_alive<0, 1>())
    .def("integrate", &PyODE::py_integrate,
        py::arg("interval"),
        py::kw_only(),
        py::arg("t_eval")=py::none(),
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0)
    .def("integrate_until", &PyODE::py_integrate_until,
        py::arg("t"),
        py::kw_only(),
        py::arg("t_eval")=py::none(),
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0)
    .def("rich_integrate", &PyODE::py_rich_integrate,
        py::arg("interval"),
        py::kw_only(),
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0)
    .def("copy", &PyODE::copy)
    .def("reset", &PyODE::reset)
    .def("clear", &PyODE::clear)
    .def("event_data", &PyODE::event_data, py::arg("event"))
    .def_property_readonly("t", &PyODE::t_array)
    .def_property_readonly("q", &PyODE::q_array)
    .def_property_readonly("event_map", &PyODE::event_map)
    .def_property_readonly("Nsys", &PyODE::Nsys)
    .def_property_readonly("runtime", &PyODE::runtime)
    .def_property_readonly("diverges", &PyODE::diverges)
    .def_property_readonly("is_dead", &PyODE::is_dead)
    .def_property_readonly("scalar_type", [](const PyODE& self){return SCALAR_TYPE[self.scalar_type];});

    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

} // PYBIND11_MODULE(ode, m)