#include "../../include/pyode/bindings/PyEvents.hpp"


using namespace ode;

PYBIND11_MODULE(events, m) {


py::class_<EventOptions>(m, "EventOpt")
    .def(py::init<const std::string&, int, bool, int>(),
            py::arg("name"),
            py::arg("max_events") = -1,
            py::arg("terminate") = false,
            py::arg("period") = 1)
    .def_readwrite("name", &EventOptions::name)
    .def_readwrite("max_events", &EventOptions::max_events)
    .def_readwrite("terminate", &EventOptions::terminate)
    .def_readwrite("period", &EventOptions::period);


py::class_<PyEvent>(m, "Event")
    .def_property_readonly("name", &PyEvent::name)
    .def_property_readonly("hides_mask", &PyEvent::hide_mask)
    .def_property_readonly("scalar_type", [](const PyEvent& self){return SCALAR_TYPE[self.scalar_type];});

py::class_<PyPrecEvent, PyEvent>(m, "PreciseEvent")
    .def(py::init<std::string, py::object, int, py::object, bool, py::object, std::string, size_t, size_t>(),
        py::arg("name"),
        py::arg("when"),
        py::arg("direction")=0,
        py::arg("mask")=py::none(),
        py::arg("hide_mask")=false,
        py::arg("event_tol")=1e-12,
        py::arg("scalar_type") = "double",
        py::arg("__Nsys")=0,
        py::arg("__Nargs")=0)
    .def_property_readonly("event_tol", &PyPrecEvent::event_tol);

py::class_<PyPerEvent, PyEvent>(m, "PeriodicEvent")
    .def(py::init<std::string, py::object, py::object, bool, std::string, size_t, size_t>(),
        py::arg("name"),
        py::arg("period"),
        py::arg("mask")=py::none(),
        py::arg("hide_mask")=false,
        py::arg("scalar_type") = "double",
        py::arg("__Nsys")=0,
        py::arg("__Nargs")=0)
    .def_property_readonly("period", &PyPerEvent::period);


} // PYBIND11_MODULE(events, m)