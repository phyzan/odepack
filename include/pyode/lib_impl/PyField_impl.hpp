#ifndef PY_FIELD_IMPL_HPP
#define PY_FIELD_IMPL_HPP

#include "../lib/PyOde.hpp"
#include "../lib/PyResult.hpp"
#include "../lib/PyField.hpp"
#include "../pycast/pycast.hpp"
#include "../../ode/Interpolation/GridInterp_impl.hpp"
#include "../../ode/Interpolation/LinearNdInterpolator_impl.hpp"

namespace ode{

// ============================================================================================
//                                      PyScalarField
// ============================================================================================


template<typename... Scalar>
double PyScalarField::operator()(Scalar... x) const{
    assert(sizeof...(Scalar) == this->ndim() && "Number of coordinates must match field dimensionality");
    Array1D<double, 0, Base::Alloc> coords = {double(x)...};
    if (!this->coords_in_bounds(coords.data())){
        return std::numeric_limits<double>::quiet_NaN();
    }
    double res;
    this->interp(&res, coords.data());
    return res;
}


//===========================================================================================
//                                      PyDelaunay
//===========================================================================================

template<size_t NDIM>
PyDelaunay<NDIM>::PyDelaunay(const py::array_t<double>& x) : PyDelaunay(parse_args(x), x) {}

template<size_t NDIM>
PyDelaunay<NDIM>::PyDelaunay(std::nullptr_t, const py::array_t<double>& x) : Base(x) {}

template<size_t NDIM>
py::object PyDelaunay<NDIM>::py_points() const{
    return py::cast(this->get_points());
}

template<size_t NDIM>
int PyDelaunay<NDIM>::py_ndim() const{
    return Base::ndim();
}

template<size_t NDIM>
int PyDelaunay<NDIM>::py_npoints() const{
    return Base::npoints();
}

template<size_t NDIM>
int PyDelaunay<NDIM>::py_nsimplices() const{
    return Base::nsimplices();
}

template<size_t NDIM>
int PyDelaunay<NDIM>::py_find_simplex(const py::array_t<double>& point) const{
    if (point.ndim() != 1 || point.shape(0) != NDIM){
        throw py::value_error("Query point must be a 1D array of length " + std::to_string(this->ndim()));
    }
    return Base::find_simplex(point.data());
}

template<size_t NDIM>
py::object PyDelaunay<NDIM>::py_get_simplices() const{
    const auto& simplices = Base::get_simplices();
    return py::cast(simplices);
}

template<size_t NDIM>
py::object PyDelaunay<NDIM>::py_get_simplex(const py::array_t<double>& point) const{
    int simplex_idx = py_find_simplex(point);
    if (simplex_idx == -1){
        return py::none();
    }
    const int* simplex = Base::get_simplex(simplex_idx);

    // Now get array of (ndim+1, ndim) containing the coordinates of the simplex vertices
    Array2D<double, 0, Base::DIM_SPX> simplex_coords(this->ndim()+1, this->ndim());
    const auto& points = Base::get_points();
    for (int i=0; i<this->ndim()+1; i++){
        const double* vertex_coords = points.ptr(simplex[i], 0);
        copy_array(simplex_coords.ptr(i, 0), vertex_coords, this->ndim());
    }
    return py::cast(simplex_coords);
}


template<size_t NDIM>
std::nullptr_t PyDelaunay<NDIM>::parse_args(const py::array_t<double>& x){
    if (x.ndim() != 1 && x.ndim() != 2){
        throw py::value_error("ScatteredField requires a 2D array for the input points (or optionally a 1D array for 1D fields)");
    }else if (x.ndim() == 2 && x.shape(0) < x.shape(1)+1){
        throw py::value_error("Number of input points must be at least one more than the number of dimensions");
    }
    return nullptr;
}



//===========================================================================================
//                                      PyScatteredField
//===========================================================================================

template<size_t NDIM>
PyScatteredField<NDIM>::PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values) : PyScatteredField(parse_args(x, values), x, values){}

template<size_t NDIM>
PyScatteredField<NDIM>::PyScatteredField(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values) : Base(x, values.data()) {}

template<size_t NDIM>
PyScatteredField<NDIM>::PyScatteredField(const PyDelaunay<NDIM>& tri, const py::array_t<double>& values) : PyScatteredField(parse_tri_args(tri, values), tri, values) {}

template<size_t NDIM>
PyScatteredField<NDIM>::PyScatteredField(std::nullptr_t, const PyDelaunay<NDIM>& tri, const py::array_t<double>& values) : Base(static_cast<const DelaunayTri<NDIM>&>(tri), values.data()) {}

template<size_t NDIM>
py::object PyScatteredField<NDIM>::py_points() const{
    return py::cast(this->get_points());
}

template<size_t NDIM>
py::object PyScatteredField<NDIM>::py_values() const{
    return py::cast(this->get_field());
}

template<size_t NDIM>
py::object PyScatteredField<NDIM>::py_delaunay_obj() const{
    return py::cast(&(this->get_delaunay()));
}


template<size_t NDIM>
double PyScatteredField<NDIM>::py_value_at(const py::args& x) const{
    constexpr Allocation Alloc = NDIM == 0 ? Allocation::Heap : Allocation::Stack;
    Array1D<double, NDIM, Alloc> point(x.size());// TODO: optimize this. Currently a temporary vector on the heap is allocated on each call when NDIM == 0
    if (x.size() != size_t(this->ndim())){
        // throw informative error including ndim of input and expected ndim
        throw py::value_error("Expected " + std::to_string(this->ndim()) + " input points, but got " + std::to_string(x.size()));
    }
    for (size_t i=0; i<x.size(); i++){
        point[i] = x[i].cast<double>();
    }
    return Base::get_value(point.data());
}

template<size_t NDIM>
std::nullptr_t PyScatteredField<NDIM>::parse_args(const py::array_t<double>& x, const py::array_t<double>& values){
    if (x.ndim() > 2){
        throw py::value_error("ScatteredField requires a 2D array for the input points (or optionally a 1D array for 1D fields)");
    }else if (x.ndim() == 1 && NDIM != 1){
        throw py::value_error("1D input points are only allowed for 1D fields");
    }else if (x.ndim() != 1 && NDIM != 0 && x.shape(1) != NDIM){
        throw py::value_error("Invalid shape for input points array. Expected shape (npoints, " + std::to_string(NDIM) + ") but got (" + std::to_string(x.shape(0)) + ", " + std::to_string(x.shape(1)) + ")");
    }else if (values.ndim() != 1){
        throw py::value_error("ScatteredField requires a 1D array for the field values");
    }else if (x.shape(0) != values.size()){
        throw py::value_error("Number of input points must match number of field values");
    }else if (x.ndim() == 2 && x.shape(0) < x.shape(1)+1){
        throw py::value_error("Number of input points must be at least one more than the number of dimensions");
    }
    return nullptr;
}

template<size_t NDIM>
std::nullptr_t PyScatteredField<NDIM>::parse_tri_args(const PyDelaunay<NDIM>& tri, const py::array_t<double>& values){
    if (values.ndim() != 1){
        throw py::value_error("ScatteredField requires a 1D array for the field values");
    }else if (tri.npoints() != values.size()){
        throw py::value_error("Number of field values must match number of points in the Delaunay triangulation");
    }
    return nullptr;
}


template<size_t NDIM>
template<typename... Scalar>
double PyScatteredField<NDIM>::operator()(Scalar... x) const{
    return Base::operator()(x...);
}


//===========================================================================================
//                                      PyVecField
//===========================================================================================




template<size_t NDIM>
PyVecField<NDIM>::PyVecField(const double* values, const std::vector<MutView1D<double>>& grid) : Base(values, grid, true) {}

template<size_t NDIM>
PyVecField<NDIM>::PyVecField(const py::array_t<double>& values, const py::args& grid) : PyVecField(values.data(), parse_args(values, grid)) {}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_x(int axis) const{
    if (axis < 0 || axis >= int(this->ndim())){
        throw py::value_error("Axis index out of bounds");
    }
    return py::cast(this->x(axis));
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_vx(int component) const{
    if (component < 0 || component >= int(this->ndim())){
        throw py::value_error("Vector component index out of bounds");
    }
    const auto& values = this->values();
    Array<double> res(values.shape(), values.ndim()-1);
    for (size_t i=0; i<res.size(); i++){
        res[i] = values[i*this->ndim() + component];
    }
    return py::cast(res);
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_coords() const{
    py::list coords;
    for (size_t i=0; i<this->ndim(); i++){
        coords.append(py::cast(this->x(i)));
    }
    return py::tuple(coords);
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_data() const{
    return py::cast(this->values());
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_value(const py::args& coords) const{
    if (coords.size() != this->ndim()){
        throw py::value_error("Expected " + std::to_string(this->ndim()) + " coordinates, but got " + std::to_string(coords.size()));
    }
    Array1D<double, NDIM, Base::Alloc> res(this->ndim());
    Array1D<double, NDIM, Base::Alloc> q(this->ndim());
    for (size_t i=0; i<coords.size(); i++){
        q[i] = coords[i].cast<double>();
    }
    check_coords(q.data());
    this->interp(res.data(), q.data());
    return py::cast(res);
}

template<size_t NDIM>
bool PyVecField<NDIM>::py_in_bounds(const py::args& coords) const{
    Array1D<double, NDIM, Base::Alloc> q(this->ndim());
    if (coords.size() != this->ndim()){
        throw py::value_error("Expected " + std::to_string(this->ndim()) + " coordinates, but got " + std::to_string(coords.size()));
    }
    for (size_t i=0; i<coords.size(); i++){
        q[i] = coords[i].cast<double>();
    }
    return this->coords_in_bounds(q.data());
}

template<size_t NDIM>
void PyVecField<NDIM>::check_coords(const double* coords) const{
    if (!this->coords_in_bounds(coords)){
        Array1D<double, NDIM, Base::Alloc> q(coords, this->ndim());
        throw py::value_error("Coordinates " + py::repr(py::cast(q)).template cast<std::string>() + " are out of bounds");
    }
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const{

    if (q0.ndim() != 1 || size_t(q0.shape(0)) != this->ndim()){
        throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(this->ndim()));
    }

    check_coords(q0.data());

    StepSequence<double> t_seq = to_step_sequence<double>(t_eval);
    try{
        double max_step_val = (max_step.is_none() ? inf<double>() : max_step.cast<double>());
        auto* result = new OdeResult<double>(this->streamline(q0.data(), length, rtol, atol, min_step, max_step_val, stepsize, direction, t_seq, method.cast<std::string>()));
        PyOdeResult py_res(result, {NDIM}, DTYPE_MAP.at("double"));
        return py::cast(py_res);
    } catch (const std::runtime_error& e){
        throw py::value_error(e.what());
    }
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const{
    if (direction != 1 && direction != -1){
        throw py::value_error("Direction must be either 1 (forward) or -1 (backward)");
    }else if (q0.ndim() != 1 || q0.shape(0) != this->ndim()){
        throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(this->ndim()));
    }

    check_coords(q0.data());

    if (normalized){
        return py::cast(PyODE(OdeData<Func<double>, void>{.rhs=Base::ode_func_norm, .obj=this}, 0., q0.data(), this->ndim(), rtol, atol, min_step, (max_step.is_none() ? inf<double>() : max_step.cast<double>()), stepsize, direction, {}, {}, method.cast<std::string>()));
    }else{
        return py::cast(PyODE(OdeData<Func<double>, void>{.rhs=Base::ode_func, .obj=this}, 0., q0.data(), this->ndim(), rtol, atol, min_step, (max_step.is_none() ? inf<double>() : max_step.cast<double>()), stepsize, direction, {}, {}, method.cast<std::string>()));
    }
}

template<size_t NDIM>
py::object PyVecField<NDIM>::py_streamplot_data(double max_length, double ds, int density) const{
    if (density <= 1){
        throw py::value_error("Density must be greater than 1");
    }
    if (max_length <= 0){
        throw py::value_error("Max length must be a positive number");
    }
    if (ds <= 0){
        throw py::value_error("ds must be a positive number");
    }

    std::vector<Array2D<double, NDIM, 0>> streamlines = this->streamplot_data(max_length, ds, size_t(density));
    py::list result;
    for (const Array2D<double, NDIM, 0>& line : streamlines){
        result.append(py::cast(line));
    }
    return result;
}

template<size_t NDIM>
std::vector<MutView1D<double>> PyVecField<NDIM>::parse_args(const py::array_t<double>& values, const py::args& py_grid){

    if (values.ndim() == 0 || py_grid.size() == 0){
        throw py::value_error("Values array and grid arrays cannot be empty");
    }else if (!(values.flags() & py::array::c_style)){
        throw py::value_error("Values array must be C-contiguous (Row major)");
    }else if (NDIM != 0 && (values.ndim() != NDIM + 1)){
        throw py::value_error("Values array must have " + std::to_string(NDIM + 1) + " dimensions");
    }else if (NDIM != 0 && py_grid.size() != NDIM){
        throw py::value_error("Number of grid arrays must match number of dimensions");
    }else if (NDIM == 0 && (py_grid.size() + 1 != size_t(values.ndim()))){
        throw py::value_error("The number of grid arrays must match the number of dimensions of the values array");
    }else if (size_t(values.shape(0)) != py_grid.size()){
        throw py::value_error("The number of vector components given was " + std::to_string(values.shape(0)) + " but expected " + std::to_string(py_grid.size()));
    }

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
            }else if (coords.size() != values.shape(axis+1)){
                // values.shape(0) is the number of components, so we check against shape(axis+1)
                throw py::value_error("Size of grid array along axis " + std::to_string(axis) + " must match the corresponding dimension of the values array");
            }
            Array1D<double> tmp(coords.data(), coords.size());
            MutView1D<double> view(tmp.release(), tmp.size()); // ownership transferred from tmp to view. MutView does not delete it however.
            grid[axis] = view;
        }catch (const py::cast_error &e) {
            throw py::value_error("Grid arrays must be 1D numpy arrays of real numbers");
        }
        axis++;
    }

    return grid;
}

}

#endif // PY_FIELD_IMPL_HPP