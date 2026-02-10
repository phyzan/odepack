#include "../../../include/odepack.hpp"
#include "../../../include/pyode/bindings_impl/PyField_impl.hpp"


namespace ode{

//===========================================================================================
//                                      PyVecField2D, 3D
//===========================================================================================

PyVecField2D::PyVecField2D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& vx, const py::array_t<double>& vy) : Base(x, y, vx, vy) {}

PyVecField3D::PyVecField3D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& z, const py::array_t<double>& vx, const py::array_t<double>& vy, const py::array_t<double>& vz) : Base(x, y, z, vx, vy, vz) {}




//===========================================================================================
//                          Explicit Template Instantiations
//===========================================================================================

// Template classes - base classes first
template class RegularGridInterpolator<double, 1>;
template class RegularGridInterpolator<double, 2>;
template class RegularGridInterpolator<double, 3>;
template class SampledVectorField<double, 2>;
template class SampledVectorField<double, 3>;

// Derived classes
template class PyScalarField<1>;
template class PyScalarField<2>;
template class PyScalarField<3>;
template class PyVecField<2>;
template class PyVecField<3>;

// Explicit constructor instantiations for PyScalarField
template PyScalarField<1>::PyScalarField(const py::array_t<double>&, const py::array_t<double>&);
template PyScalarField<2>::PyScalarField(const py::array_t<double>&, const py::array_t<double>&, const py::array_t<double>&);
template PyScalarField<3>::PyScalarField(const py::array_t<double>&, const py::array_t<double>&, const py::array_t<double>&, const py::array_t<double>&);

// Explicit instantiations for PyScalarField::operator() member function template
// These are needed because operator() is a template member function that gets called
// from dynamically compiled ODE code
template double PyScalarField<1>::operator()(double) const;
template double PyScalarField<2>::operator()(double, double) const;
template double PyScalarField<3>::operator()(double, double, double) const;


// PyVecField<2> template member functions
template py::object PyVecField<2>::py_x<0>() const;
template py::object PyVecField<2>::py_x<1>() const;
template py::object PyVecField<2>::py_vx<0>() const;
template py::object PyVecField<2>::py_vx<1>() const;
template py::object PyVecField<2>::py_vx_at<0, double, double>(double, double) const;
template py::object PyVecField<2>::py_vx_at<1, double, double>(double, double) const;
template py::object PyVecField<2>::py_vector<double, double>(double, double) const;
template bool PyVecField<2>::py_in_bounds<double, double>(double, double) const;

// PyVecField<3> template member functions
template py::object PyVecField<3>::py_x<0>() const;
template py::object PyVecField<3>::py_x<1>() const;
template py::object PyVecField<3>::py_x<2>() const;
template py::object PyVecField<3>::py_vx<0>() const;
template py::object PyVecField<3>::py_vx<1>() const;
template py::object PyVecField<3>::py_vx<2>() const;
template py::object PyVecField<3>::py_vx_at<0, double, double, double>(double, double, double) const;
template py::object PyVecField<3>::py_vx_at<1, double, double, double>(double, double, double) const;
template py::object PyVecField<3>::py_vector<double, double, double>(double, double, double) const;
template bool PyVecField<3>::py_in_bounds<double, double, double>(double, double, double) const;

// PyScatteredField::operator() instantiations for dynamically compiled ODEs

#define INST_PY_TRI(N) \
template PyDelaunay<N>::PyDelaunay(const py::array_t<double>& x);\
template py::object PyDelaunay<N>::py_points() const;\
template int PyDelaunay<N>::py_ndim() const;\
template int PyDelaunay<N>::py_npoints() const;\
template int PyDelaunay<N>::py_nsimplices() const;\
template int PyDelaunay<N>::py_find_simplex(const py::array_t<double>&) const;\
template py::object PyDelaunay<N>::py_get_simplex(const py::array_t<double>& point) const;\
template py::object PyDelaunay<N>::py_get_simplices() const;\

INST_PY_TRI(0)
INST_PY_TRI(1)
INST_PY_TRI(2)
INST_PY_TRI(3)


// For NDIM = 0:
#define INST_PY_SCAT_FIELD(N)\
template PyScatteredField<N>::PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values);\
template PyScatteredField<N>::PyScatteredField(const PyDelaunay<N>& tri, const py::array_t<double>& values);\
template py::object PyScatteredField<N>::py_points() const;\
template py::object PyScatteredField<N>::py_values() const;\
template py::object PyScatteredField<N>::py_delaunay_obj() const;\
template double PyScatteredField<N>::py_value_at(const py::args& x) const;\


INST_PY_SCAT_FIELD(0)
template double PyScatteredField<0>::operator()(double) const;
template double PyScatteredField<0>::operator()(double, double) const;
template double PyScatteredField<0>::operator()(double, double, double) const;

INST_PY_SCAT_FIELD(1)
template double PyScatteredField<1>::operator()(double) const;

INST_PY_SCAT_FIELD(2)
template double PyScatteredField<2>::operator()(double, double) const;

INST_PY_SCAT_FIELD(3)
template double PyScatteredField<3>::operator()(double, double, double) const;


} // namespace ode