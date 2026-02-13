#include "../../../include/pyode/lib_impl/PyField_impl.hpp"
#include "../../../include/ode/Interpolation/SampledVectorfields_impl.hpp"


namespace ode{


PyScalarField::PyScalarField(const py::array_t<double>& values, const py::args& py_grid) : PyScalarField(values.data(), parse_args(values, py_grid, 0)) {}
PyScalarField::PyScalarField(const py::array_t<double>& values, const py::args& py_grid, size_t NDIM) : PyScalarField(values.data(), parse_args(values, py_grid, NDIM)) {}
PyScalarField::PyScalarField(const double* values, const std::vector<MutView1D<double>>& grid) : Base(values, 1, grid) {}

double PyScalarField::py_value_at(const py::args& x) const{
    if (x.size() != this->ndim()){
        throw py::value_error("Expected " + std::to_string(this->ndim()) + " coordinates, but got " + std::to_string(x.size()));
    }
    Array1D<double, 0, Base::Alloc> coords(x.size());
    for (size_t i=0; i<x.size(); i++){
        try {
            coords[i] = x[i].cast<double>();
        }catch (const py::cast_error &e) {
            throw py::value_error("All coordinates must be real numbers");
        }
    }
    if (!this->coords_in_bounds(coords.data())){
        throw py::value_error("Coordinates out of bounds");
    }
    double res;
    this->interp(&res, coords.data());
    return res;
}

std::vector<MutView1D<double>> PyScalarField::parse_args(const py::array_t<double>& values, const py::args& py_grid, size_t NDIM){
    if (py_grid.size() == 0){
        throw py::value_error("Grid arrays cannot be empty");
    }else if (NDIM != 0 && py_grid.size() != NDIM){
        throw py::value_error("Number of grid arrays must match number of dimensions");
    }else if (NDIM == 0 && py_grid.size() != size_t(values.ndim())){
        throw py::value_error("The number of grid arrays must match the number of dimensions of the values array");
    }

    size_t expected_vals = prod(values.shape(), values.ndim());
    size_t grid_points = 1;
    std::vector<MutView1D<double>> grid(py_grid.size());
    size_t axis = 0;
    for (const py::handle& item : py_grid){
        try {
            auto coords = item.cast<py::array_t<double>>();
            if (coords.ndim() != 1){
                throw py::value_error("Grid arrays must be 1D");
            }else if (!is_sorted(coords)){
                throw py::value_error("Grid arrays must be sorted in ascending order");
            }else if (coords.size() < 2){
                throw py::value_error("Grid arrays must have at least 2 points");
            }else if (coords.size() != values.shape(axis)){
                throw py::value_error("Size of grid array along axis " + std::to_string(axis) + " must match the corresponding dimension of the values array");
            }
            Array1D<double> tmp(coords.data(), coords.size());
            MutView1D<double> view(tmp.release(), tmp.size()); // ownership transferred from tmp to view. MutView does not delete it however.
            grid[axis] = view;
            grid_points *= coords.size();
        }catch (const py::cast_error &e) {
            throw py::value_error("Grid arrays must be 1D numpy arrays of real numbers");
        }
        axis++;
    }

    if (grid_points != expected_vals){
        throw py::value_error("The total number of grid points (product of grid array sizes) must match the size of the values array");
    }
    return grid;
}

PyScalarField1D::PyScalarField1D(const py::array_t<double>& values, const py::args& py_grid) : Base(values, py_grid, 1) {}
PyScalarField2D::PyScalarField2D(const py::array_t<double>& values, const py::args& py_grid) : Base(values, py_grid, 2) {}
PyScalarField3D::PyScalarField3D(const py::array_t<double>& values, const py::args& py_grid) : Base(values, py_grid, 3) {}


template class PyVecField<0>;

py::object PyVecField2D::x() const{ return Base::py_x(0); }
py::object PyVecField2D::y() const{ return Base::py_x(1); }
py::object PyVecField2D::vx() const{ return Base::py_vx(0); }
py::object PyVecField2D::vy() const{ return Base::py_vx(1); }

py::object PyVecField3D::x() const{ return Base::py_x(0); }
py::object PyVecField3D::y() const{ return Base::py_x(1); }
py::object PyVecField3D::z() const{ return Base::py_x(2); }
py::object PyVecField3D::vx() const{ return Base::py_vx(0); }
py::object PyVecField3D::vy() const{ return Base::py_vx(1); }
py::object PyVecField3D::vz() const{ return Base::py_vx(2); }

//===========================================================================================
//                          Explicit Template Instantiations
//===========================================================================================

// Explicit instantiations for PyScalarField::operator() member function template
// These are needed because operator() is a template member function that gets called
// from dynamically compiled ODE code
double PyScalarField1D::operator()(double x) const{
    return Base::operator()(x);
}
double PyScalarField2D::operator()(double x, double y) const{
    return Base::operator()(x, y);
}

double PyScalarField3D::operator()(double x, double y, double z) const{
    return Base::operator()(x, y, z);
}


#define INST_PY_TRI(N) \
template PyDelaunay<N>::PyDelaunay(const py::array_t<double>& x);\
template py::object PyDelaunay<N>::py_points() const;\
template int PyDelaunay<N>::py_ndim() const;\
template int PyDelaunay<N>::py_npoints() const;\
template int PyDelaunay<N>::py_nsimplices() const;\
template int PyDelaunay<N>::py_find_simplex(const py::array_t<double>&) const;\
template py::object PyDelaunay<N>::py_get_simplex(const py::array_t<double>& point) const;\
template py::object PyDelaunay<N>::py_get_simplices() const;\
template py::dict PyDelaunay<N>::py_get_state() const;\
template PyDelaunay<N> PyDelaunay<N>::py_set_state(const py::dict& state);\

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