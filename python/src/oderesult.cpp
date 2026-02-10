#include "../../include/pyode/bindings/PyResult.hpp"


using namespace ode;

PYBIND11_MODULE(oderesult, m) {

py::class_<PyOdeResult>(m, "OdeResult")
    .def(py::init<PyOdeResult>(), py::arg("result"))
    .def_property_readonly("t", &PyOdeResult::t)
    .def_property_readonly("q", &PyOdeResult::q)
    .def_property_readonly("event_map", &PyOdeResult::event_map)
    .def("event_data", &PyOdeResult::event_data, py::arg("event"))
    .def_property_readonly("diverges", &PyOdeResult::diverges)
    .def_property_readonly("success", &PyOdeResult::success)
    .def_property_readonly("runtime", &PyOdeResult::runtime)
    .def_property_readonly("message", &PyOdeResult::message)
    .def("examine", &PyOdeResult::examine)
    .def_property_readonly("scalar_type", [](const PyOdeResult& self){return SCALAR_TYPE[self.scalar_type];});

py::class_<PyOdeSolution, PyOdeResult>(m, "OdeSolution")
    .def(py::init<PyOdeSolution>(), py::arg("result"))
    .def("__call__", [](const PyOdeSolution& self, const py::object& t){return self(t);});

} // PYBIND11_MODULE(oderesult, m)