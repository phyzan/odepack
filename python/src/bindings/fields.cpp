#include "../../../include/pyode/lib/PyField.hpp"

// namespace ode{

using namespace ode;

PYBIND11_MODULE(fields, m) {


py::class_<PyScalarField>(m, "SampledScalarField")
    .def(py::init<py::array_t<double>, py::args>(),
        py::arg("values"))
    .def("__call__", &PyScalarField::py_value_at); //takes N coordinates as separate arguments, returns scalar value at that point;

py::class_<PyScalarField1D, PyScalarField>(m, "SampledScalarField1D")
    .def(py::init<py::array_t<double>, py::args>(),
         py::arg("values"));

py::class_<PyScalarField2D, PyScalarField>(m, "SampledScalarField2D")
    .def(py::init<py::array_t<double>, py::args>(),
         py::arg("values"));

py::class_<PyScalarField3D, PyScalarField>(m, "SampledScalarField3D")
    .def(py::init<py::array_t<double>, py::args>(),
         py::arg("values"));



py::class_<PyVecField<0>>(m, "SampledVectorField")
    .def(py::init<py::array_t<double>, py::args>(),
        py::arg("values"))
    .def("coords", &PyVecField<0>::py_coords) //tuple of coordinate arrays
    .def("values", &PyVecField<0>::py_data) //(N+1)-dimensional : shape (nx, ny, ..., ndim)
    .def("__call__", &PyVecField<0>::py_value) //takes N coordinates as separate arguments, returns vector value at that point
    .def("in_bounds", &PyVecField<0>::py_in_bounds) //takes N coordinates as separate arguments, returns True if point is in bounds of the field
    .def("streamline", &PyVecField<0>::py_streamline,
        py::arg("q0"),
        py::arg("length"),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("t_eval")=py::none(),
        py::arg("method")="RK45"
    )
    .def("get_ode", &PyVecField<0>::py_streamline_ode,
        py::arg("q0"),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=py::none(),
        py::arg("stepsize")=0.,
        py::arg("direction")=1,
        py::arg("method")="RK45",
        py::arg("normalized")=false, py::keep_alive<0, 1>()
    )
    .def("streamplot_data", &PyVecField<0>::py_streamplot_data,
        py::arg("max_length"),
        py::arg("ds"),
        py::arg("density")=30
    );

py::class_<PyVecField2D, PyVecField<0>>(m, "SampledVectorField2D")
    .def(py::init<py::array_t<double>, py::args>(),
        py::arg("values"))
    .def_property_readonly("x", &PyVecField2D::x)
    .def_property_readonly("y", &PyVecField2D::y)
    .def_property_readonly("vx", &PyVecField2D::vx)
    .def_property_readonly("vy", &PyVecField2D::vy);

py::class_<PyVecField3D, PyVecField<0>>(m, "SampledVectorField3D")
    .def(py::init<py::array_t<double>, py::args>(),
        py::arg("values"))
    .def_property_readonly("x", &PyVecField3D::x)
    .def_property_readonly("y", &PyVecField3D::y)
    .def_property_readonly("z", &PyVecField3D::z)
    .def_property_readonly("vx", &PyVecField3D::vx)
    .def_property_readonly("vy", &PyVecField3D::vy)
    .def_property_readonly("vz", &PyVecField3D::vz);


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
        py::arg("points"))
    .def(py::pickle([](const PyDelaunay<0>& obj){ return obj.py_get_state(); },
        &PyDelaunay<0>::py_set_state));

py::class_<PyDelaunay<1>, PyDelaunayBase>(m, "DelaunayTri1D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"))
    .def(py::pickle([](const PyDelaunay<1>& obj){ return obj.py_get_state(); },
                    &PyDelaunay<1>::py_set_state));

py::class_<PyDelaunay<2>, PyDelaunayBase>(m, "DelaunayTri2D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"))
    .def(py::pickle([](const PyDelaunay<2>& obj){ return obj.py_get_state(); },
                    &PyDelaunay<2>::py_set_state));

py::class_<PyDelaunay<3>, PyDelaunayBase>(m, "DelaunayTri3D")
    .def(py::init<py::array_t<double>>(),
        py::arg("points"))
    .def(py::pickle([](const PyDelaunay<3>& obj){ return obj.py_get_state(); },
                    &PyDelaunay<3>::py_set_state));


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
