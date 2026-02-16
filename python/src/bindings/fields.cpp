#include "../../../include/pyode/lib/PyField.hpp"

// namespace ode{

using namespace ode;

PYBIND11_MODULE(fields, m) {


py::class_<PyScalarField>(m, "SampledScalarField")
    .def("__call__", &PyScalarField::py_value_at); //takes N coordinates as separate arguments, returns scalar value at that point;



py::class_<VirtualVectorField>(m, "SampledVectorField")
    // .def("coords", &PyVecField::py_coords) //tuple of coordinate arrays
    // .def("values", &PyVecField::py_data) //(N+1)-dimensional : shape (nx, ny, ..., ndim)
    .def("__call__", &PyNdInterp::py_value_at) //takes N coordinates as separate arguments, returns vector value at that point
    // .def("in_bounds", &PyVecField::py_in_bounds) //takes N coordinates as separate arguments, returns True if point is in bounds of the field
    .def("streamline", &PyVecField::py_streamline,
        py::arg("q0"),
        py::arg("length"),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("t_eval")=py::none(),
        py::arg("method")="RK45",
        py::arg("normalized")=true
    )
    .def("get_ode", &PyVecField::py_streamline_ode,
        py::arg("q0"),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("method")="RK45",
        py::arg("normalized")=false, py::keep_alive<0, 1>()
    );

py::class_<PyRegScalarField, PyScalarField, RegularGridInterpolator<double, 0, true>>(m, "RegularGridScalarField")
    .def(py::init<py::array_t<double>, py::args>(),
        py::arg("values"));


py::class_<RegularVectorField<double, 0, true>, VirtualVectorField, RegularGridInterpolator<double, 0, true>>(m, "RegularGridVectorField")
    .def(py::init(&PyRegVecField::init_main),
        py::arg("values"))
    .def("component", &PyRegVecField::component,
        py::arg("i"))
    .def("streamplot_data", &PyRegVecField::py_streamplot_data,
        py::arg("max_length"),
        py::arg("ds"),
        py::arg("density")=30
    );
        
py::class_<PyScatteredField, PyScalarField, ScatteredNdInterpolator<0, true>>(m, "ScatteredScalarField")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("points"),
        py::arg("values"))
    .def(py::init<PyDelaunay, py::array_t<double>>(),
        py::arg("tri"),
        py::arg("values"))
    .def_property_readonly("points", &PyScatteredField::py_points)
    .def_property_readonly("values", &PyScatteredField::py_values)
    .def_property_readonly("tri", &PyScatteredField::py_delaunay);

py::class_<ScatteredVectorField<0, true>, VirtualVectorField, ScatteredNdInterpolator<0, true>>(m, "ScatteredVectorField")
    .def(py::init(&PyScatVecField::init),
        py::arg("points"),
        py::arg("values"))
    .def(py::init(&PyScatVecField::init_tri),
        py::arg("tri"),
        py::arg("values"));


} // namespace ode
