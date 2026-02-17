#include "../../../include/pyode/lib/PyInterp.hpp"


using namespace ode;



PYBIND11_MODULE(interp, m) {

py::class_<VirtualNdInterpolator>(m, "NdInterpolator")
    .def_property_readonly("ndim", &PyNdInterp::ndim)
    .def("__call__", &PyNdInterp::py_value_at);

py::class_<RegularGridInterpolator<double, 0, true>, VirtualNdInterpolator>(m, "RegularGridInterpolator")
    .def(py::init(&PyRegGridInterp::init_main),
        py::arg("values"))
    .def_property_readonly("values", &PyRegGridInterp::get_values)
    .def_property_readonly("grid", &PyRegGridInterp::get_grid);

py::class_<PyDelaunay>(m, "DelaunayTri")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"))
    .def(py::pickle([](const PyDelaunay& obj){ return obj.py_get_state(); },
        &PyDelaunay::py_set_state))
    .def_property_readonly("ndim", &PyDelaunay::py_ndim)
    .def_property_readonly("npoints", &PyDelaunay::py_npoints)
    .def_property_readonly("nsimplices", &PyDelaunay::py_nsimplices)
    .def_property_readonly("points", &PyDelaunay::py_points)
    .def_property_readonly("simplices", &PyDelaunay::py_get_simplices)
    .def_property_readonly("total_volume", &PyDelaunay::total_volume)
    .def("find_simplex", &PyDelaunay::py_find_simplex, py::arg("coords"))
    .def("get_simplex", &PyDelaunay::py_get_simplex,
        py::arg("coords"));


py::class_<ScatteredNdInterpolator<0, true>, VirtualNdInterpolator>(m, "ScatteredNdInterpolator")
    .def(py::init(&PyScatteredInterp::init_main),
        py::arg("points"), py::arg("values"))
    .def(py::init(&PyScatteredInterp::init_tri),
        py::arg("tri"), py::arg("values"))
    .def_property_readonly("points", &PyScatteredInterp::py_points)
    .def_property_readonly("values", &PyScatteredInterp::py_values)
    .def_property_readonly("tri", &PyScatteredInterp::py_delaunay_obj);




} // PYBIND11_MODULE