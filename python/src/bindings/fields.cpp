#include "../../../include/pyode/lib/PyField.hpp"

// namespace ode{

using namespace ode;

PYBIND11_MODULE(fields, m) {


py::class_<PyVecFieldBase>(m, "SampledVectorField")
    .def("streamline", &PyVecFieldBase::py_streamline,
        py::arg("q0"),
        py::arg("length"),
        py::arg("rtol")=1e-12,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("t_eval")=py::none(),
        py::arg("method")="RK45"
    )
    .def("get_ode", &PyVecFieldBase::py_streamline_ode,
        py::arg("q0"),
        py::arg("rtol")=1e-12,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("method")="RK45",
        py::arg("normalized")=false, py::keep_alive<0, 1>()
    )
    .def("streamplot_data", &PyVecFieldBase::py_streamplot_data,
        py::arg("max_length"),
        py::arg("ds"),
        py::arg("density")=30
    );

py::class_<PyVecField2D, PyVecFieldBase>(m, "SampledVectorField2D")
    .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
        py::arg("x"),
        py::arg("y"),
        py::arg("vx"),
        py::arg("vy"))
    .def_property_readonly("x", &PyVecField2D::py_x<0>)
    .def_property_readonly("y", &PyVecField2D::py_x<1>)
    .def_property_readonly("vx", &PyVecField2D::py_vx<0>)
    .def_property_readonly("vy", &PyVecField2D::py_vx<1>)
    .def("get_vx", &PyVecField2D::py_vx_at<0, double, double>, py::arg("x"), py::arg("y"))
    .def("get_vy", &PyVecField2D::py_vx_at<1, double, double>, py::arg("x"), py::arg("y"))
    .def("__call__", &PyVecField2D::py_vector<double, double>, py::arg("x"), py::arg("y"))
    .def("in_bounds", &PyVecField2D::py_in_bounds<double, double>, py::arg("x"), py::arg("y"));

py::class_<PyVecField3D, PyVecFieldBase>(m, "SampledVectorField3D")
    .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("vx"),
        py::arg("vy"),
        py::arg("vz"))
    .def_property_readonly("x", &PyVecField3D::py_x<0>)
    .def_property_readonly("y", &PyVecField3D::py_x<1>)
    .def_property_readonly("z", &PyVecField3D::py_x<2>)
    .def_property_readonly("vx", &PyVecField3D::py_vx<0>)
    .def_property_readonly("vy", &PyVecField3D::py_vx<1>)
    .def_property_readonly("vz", &PyVecField3D::py_vx<2>)
    .def("get_vx", &PyVecField3D::py_vx_at<0, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
    .def("get_vy", &PyVecField3D::py_vx_at<1, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
    .def("__call__", &PyVecField3D::py_vector<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
    .def("in_bounds", &PyVecField3D::py_in_bounds<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"));



py::class_<PyScalarField<1>>(m, "SampledScalarField1D")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("f"),
        py::arg("x"));

py::class_<PyScalarField<2>>(m, "SampledScalarField2D")
    .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
        py::arg("f"),
        py::arg("x"),
        py::arg("y"));

py::class_<PyScalarField<3>>(m, "SampledScalarField3D")
    .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
        py::arg("f"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"));

py::class_<PyDelaunayBase>(m, "DelaunayTriangulation")
    .def_property_readonly("ndim", &PyDelaunayBase::py_ndim)
    .def_property_readonly("npoints", &PyDelaunayBase::py_npoints)
    .def_property_readonly("nsimplices", &PyDelaunayBase::py_nsimplices)
    .def_property_readonly("points", &PyDelaunayBase::py_points)
    .def_property_readonly("simplices", &PyDelaunayBase::py_get_simplices)
    .def("find_simplex", &PyDelaunayBase::py_find_simplex, py::arg("coords"))
    .def("get_simplex", &PyDelaunayBase::py_get_simplex,
        py::arg("coords"));

py::class_<PyDelaunay<0>, PyDelaunayBase>(m, "DelaunayTriND")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"));

py::class_<PyDelaunay<1>, PyDelaunayBase>(m, "DelaunayTri1D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"));

py::class_<PyDelaunay<2>, PyDelaunayBase>(m, "DelaunayTri2D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"));

py::class_<PyDelaunay<3>, PyDelaunayBase>(m, "DelaunayTri3D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"));


py::class_<PyScatteredField<0>>(m, "ScatteredScalarFieldND")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("points"),
        py::arg("values"))
    .def(py::init<PyDelaunay<0>, py::array_t<double>>(),
        py::arg("tri"),
        py::arg("values"))
    .def_property_readonly("points", &PyScatteredField<0>::py_points)
    .def_property_readonly("values", &PyScatteredField<0>::py_values)
    .def_property_readonly("tri", &PyScatteredField<0>::py_delaunay_obj)
    .def("__call__", &PyScatteredField<0>::py_value_at);

py::class_<PyScatteredField<1>>(m, "ScatteredScalarField1D")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("points"),
        py::arg("values"))
    .def(py::init<PyDelaunay<1>, py::array_t<double>>(),
        py::arg("tri"),
        py::arg("values"))
    .def_property_readonly("points", &PyScatteredField<1>::py_points)
    .def_property_readonly("values", &PyScatteredField<1>::py_values)
    .def_property_readonly("tri", &PyScatteredField<1>::py_delaunay_obj)
    .def("__call__", &PyScatteredField<1>::py_value_at);

py::class_<PyScatteredField<2>>(m, "ScatteredScalarField2D")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("points"),
        py::arg("values"))
    .def(py::init<PyDelaunay<2>, py::array_t<double>>(),
        py::arg("tri"),
        py::arg("values"))
    .def_property_readonly("points", &PyScatteredField<2>::py_points)
    .def_property_readonly("values", &PyScatteredField<2>::py_values)
    .def_property_readonly("tri", &PyScatteredField<2>::py_delaunay_obj)
    .def("__call__", &PyScatteredField<2>::py_value_at);

py::class_<PyScatteredField<3>>(m, "ScatteredScalarField3D")
    .def(py::init<py::array_t<double>, py::array_t<double>>(),
        py::arg("points"),
        py::arg("values"))
    .def(py::init<PyDelaunay<3>, py::array_t<double>>(),
        py::arg("tri"),
        py::arg("values"))
    .def_property_readonly("points", &PyScatteredField<3>::py_points)
    .def_property_readonly("values", &PyScatteredField<3>::py_values)
    .def_property_readonly("tri", &PyScatteredField<3>::py_delaunay_obj)
    .def("__call__", &PyScatteredField<3>::py_value_at);


} // namespace ode
