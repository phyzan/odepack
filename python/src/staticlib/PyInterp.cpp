#include "../../../include/pyode/lib/PyInterp.hpp"
#include "../../../include/ode/Interpolation/NdInterpolator_impl.hpp"
#include "../../../include/ode/Interpolation/Regular/Grids_impl.hpp"
#include "../../../include/ode/Interpolation/Regular/RegularGridInterpolator_impl.hpp"
#include "../../../include/ode/Interpolation/Scattered/ScatteredNdInterpolator_impl.hpp"
#include "../../../include/ode/Interpolation/Scattered/Delaunay_impl.hpp"
#include "../../../include/pyode/pycast/pycast.hpp"

namespace ode{


//Explicit instanciation

template class RegularGridInterpolator<double, 0, true>;
template class RegularGrid<double, 0>;
template class ScatteredNdInterpolator<0, true>;
template class DelaunayTri<0>;

//===========================================================================================
//                                      PyNdInterp
//===========================================================================================

py::object PyNdInterp::py_value_at(const VirtualNdInterpolator& self, const py::args& x){
    if (x.size() != size_t(self.ndim())){
        throw py::value_error("Expected " + std::to_string(self.ndim()) + " coordinates, but got " + std::to_string(x.size()));
    }
    // TODO: optimize by avoiding temporary allocations
    Array1D<double, 0> coords(x.size());
    Array<double> out(nullptr, self.output_shape(), self.output_dims());
    for (size_t i=0; i<x.size(); i++){
        try {
            coords[i] = x[i].cast<double>();
        }catch (const py::cast_error &e) {
            throw py::value_error("All coordinates must be real numbers");
        }
    }
    
    if (!self.interp(out.data(), coords.data())){
        throw py::value_error("Coordinates out of bounds");
    }
    if (out.size() == 1){
        return py::float_(out[0]);
    }
    else{
        return py::cast(out);
    }
}

int PyNdInterp::ndim(const VirtualNdInterpolator& self) {
    return self.ndim();
}

//===========================================================================================
//                                      PyRegGridInterp
//===========================================================================================

PyRegGridInterp::CLS PyRegGridInterp::init_main(const py::array_t<double>& values, const py::args& py_grid) {
    return init(values, parse_args(values, py_grid, true), true);
}

py::object PyRegGridInterp::get_values(const CLS& self) {
    return py::cast(self.values());
}

PyRegGridInterp::CLS PyRegGridInterp::init(const py::array_t<double>& values, const std::vector<Array1D<double>>& grid, bool coord_axis_first) {
    //create a proper view over the data first.
    const double* data = values.data();
    View<double> v(data, values.size());
    if (coord_axis_first){
        // current shape is for e.g. 2D: (nx, ny, ...), where ... is any shape
        // we need to convert this to (n_points, ...)
        // github copilot complete this
        std::vector<int> new_shape(values.ndim() - grid.size()+1, 1); //fill with 1's initially
        int n_points = get_point_count(grid);
        new_shape[0] = n_points;
        for (int i = 1; i < int(new_shape.size()); i++){
            new_shape[i] = int(values.shape(i + grid.size() - 1));
        }
        v.reshape(new_shape.data(), new_shape.size());
    }else{
        // current shape is e.g. for 2D: (..., nx, ny), where ... is any shape
        // we need to convert this to (..., n_points)
        std::vector<int> new_shape(values.ndim() - grid.size()+1, 1); //fill with 1's initially
        int n_points = get_point_count(grid);
        for (int i = 0; i < int(new_shape.size()) - 1; i++){
            new_shape[i] = int(values.shape(i));
        }
        new_shape.back() = n_points;
        v.reshape(new_shape.data(), new_shape.size());
    }
    return {v, grid, coord_axis_first};
}

PyRegGridInterp::CLS PyRegGridInterp::init(const py::array_t<double>& values, const py::args& py_grid, bool coord_axis_first) {
    return init(values, parse_args(values, py_grid, coord_axis_first), coord_axis_first);
}



std::vector<Array1D<double>> PyRegGridInterp::parse_args(const py::array_t<double>& values, const py::args& py_grid, bool coord_axis_first){
    if (py_grid.size() == 0){
        throw py::value_error("Grid arrays cannot be empty");
    }else if (py_grid.size() > size_t(values.ndim())){
        throw py::value_error("The values array must have at least as many dimensions as the number of grid arrays");
    }else if (!(values.flags() & py::array::c_style)){
        throw py::value_error("Values array must be C-contiguous");
    }

    std::vector<Array1D<double>> grid(py_grid.size());
    size_t axis = 0;
    for (const py::handle& item : py_grid){
        try {
            auto coords = item.cast<py::array_t<double>>();
            if (coords.ndim() != 1){
                throw py::value_error("Grid arrays must be 1D");
            }else if (!isStrictlyAscending(coords.data(), coords.size())){
                throw py::value_error("Grid arrays must be sorted in ascending order");
            }else if (coords.size() < 2){
                throw py::value_error("Grid arrays must have at least 2 points");
            }else if (size_t(values.ndim()) > py_grid.size()){
                const int offset = coord_axis_first ? 0 : int(values.ndim() - py_grid.size());
                if (coords.size() != values.shape(int(axis) + offset)){
                    throw py::value_error("Size of grid array along axis " + std::to_string(axis) + " must match the corresponding dimension of the values array");
                }
            }
            if (size_t(values.ndim()) == py_grid.size() && coords.size() != values.shape(axis)){
                throw py::value_error("Size of grid array along axis " + std::to_string(axis) + " must match the corresponding dimension of the values array");
            }
            grid[axis] = Array1D<double>(coords.data(), coords.size());
        }catch (const py::cast_error &e) {
            throw py::value_error("Grid arrays must be 1D numpy arrays of real numbers");
        }
        axis++;
    }
    return grid;
}



//===========================================================================================
//                                      PyDelaunay
//===========================================================================================

PyDelaunay::PyDelaunay(const py::array_t<double>& x) : PyDelaunay(parse_args(x), x) {}

PyDelaunay::PyDelaunay(TriPtr<0> tri) : tri_(std::move(tri)) {}

PyDelaunay::PyDelaunay(std::nullptr_t, const py::array_t<double>& x) : tri_(std::make_shared<DelaunayTri<0>>(x.data(), x.shape(0), (x.ndim() == 1 ? 1 : x.shape(1)))) {}

py::object PyDelaunay::py_points() const{
    return py::cast(tri_->get_points());
}

int PyDelaunay::py_ndim() const{
    return tri_->ndim();
}

int PyDelaunay::py_npoints() const{
    return tri_->npoints();
}

int PyDelaunay::py_nsimplices() const{
    return tri_->nsimplices();
}

int PyDelaunay::py_find_simplex(const py::array_t<double>& point) const{
    if (point.ndim() != 1 || point.shape(0) != tri_->ndim()){
        throw py::value_error("Query point must be a 1D array of length " + std::to_string(tri_->ndim()));
    }
    return tri_->find_simplex(point.data());
}


py::object PyDelaunay::py_get_simplices() const{
    const auto& simplices = tri_->get_simplices();
    return py::cast(simplices);
}

py::object PyDelaunay::py_get_simplex(const py::array_t<double>& point) const{
    int simplex_idx = py_find_simplex(point);
    if (simplex_idx == -1){
        return py::none();
    }
    const int* simplex = tri_->get_simplex(simplex_idx);

    // Now get array of (ndim+1, ndim) containing the coordinates of the simplex vertices
    Array2D<double, 0, Base::DIM_SPX> simplex_coords(tri_->ndim()+1, tri_->ndim());
    const auto& points = tri_->get_points();
    for (int i=0; i<tri_->ndim()+1; i++){
        const double* vertex_coords = points.ptr(simplex[i], 0);
        copy_array(simplex_coords.ptr(i, 0), vertex_coords, tri_->ndim());
    }
    return py::cast(simplex_coords);
}

TriPtr<0> PyDelaunay::tri() const{
    return tri_;
}


std::nullptr_t PyDelaunay::parse_args(const py::array_t<double>& x){
    if (x.ndim() != 1 && x.ndim() != 2){
        throw py::value_error("ScatteredField requires a 2D array for the input points (or optionally a 1D array for 1D fields)");
    }else if (x.ndim() == 2 && x.shape(0) < x.shape(1)+1){
        throw py::value_error("Number of input points must be at least one more than the number of dimensions");
    }
    return nullptr;
}

py::dict PyDelaunay::py_get_state() const{
    const auto& points = tri_->get_points();
    const auto& simplices = tri_->get_simplices();
    const auto& neighbors = tri_->get_neighbors();
    const auto& v0 = tri_->get_v0();
    const auto& invT = tri_->get_invT();

    py::dict state;
    state["points"] = py::cast(points);
    state["simplices"] = py::cast(simplices);
    state["neighbors"] = py::cast(neighbors);
    state["v0"] = py::cast(v0);
    state["invT"] = py::cast(invT);
    return state;
}


PyDelaunay PyDelaunay::py_set_state(const py::dict& state){
    try {
        //Array2D<double, 0, NDIM>
        const auto& py_points = state["points"].cast<py::array_t<double>>(); 

        // Array2D<int, 0, Base::DIM_SPX>
        const auto& py_simplices = state["simplices"].cast<py::array_t<int>>();

        // Array2D<int, 0, Base::DIM_SPX>
        const auto& py_neighbors = state["neighbors"].cast<py::array_t<int>>();

        // Array2D<double, 0, NDIM>
        const auto& py_v0 = state["v0"].cast<py::array_t<double>>();

        //Array3D<double, 0, NDIM, NDIM>
        const auto& py_invT = state["invT"].cast<py::array_t<double>>();

        PyDelaunay obj;
        
        View2D<double, 0, 0> points(py_points.data(), py_points.shape(), py_points.ndim());
        View2D<int, 0, 0> simplices(py_simplices.data(), py_simplices.shape(), py_simplices.ndim());
        View2D<int, 0, 0> neighbors(py_neighbors.data(), py_neighbors.shape(), py_neighbors.ndim());
        View2D<double, 0, 0> v0(py_v0.data(), py_v0.shape(), py_v0.ndim());
        View3D<double, 0, 0, 0> invT(py_invT.data(), py_invT.shape(), py_invT.ndim());
        
        obj.tri_->set_state(points, simplices, neighbors, v0, invT);
        return obj;
    }catch (const py::cast_error &e) {
        throw py::value_error(e.what());
    }
}


//===========================================================================================
//                                      PyScatteredInterp
//===========================================================================================

PyScatteredInterp::CLS PyScatteredInterp::init_main(const py::array_t<double>& x, const py::array_t<double>& values) {
    return init(parse_args(x, values, true), x, values, true);
}



PyScatteredInterp::CLS PyScatteredInterp::init_tri(const PyDelaunay& tri, const py::array_t<double>& values) {
    return init(parse_tri_args(tri, values, true), tri, values, true);
}

PyScatteredInterp::CLS PyScatteredInterp::init(const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first) {
    return init(parse_args(x, values, coord_axis_first), x, values, coord_axis_first);
}

PyScatteredInterp::CLS PyScatteredInterp::init(const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first) {
    return init(parse_tri_args(tri, values, coord_axis_first), tri, values, coord_axis_first);
}


PyScatteredInterp::CLS PyScatteredInterp::init(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first) {
    const int ndim = (x.ndim() == 1) ? 1 : int(x.shape(1));
    return {x.data(), values, ndim, coord_axis_first};
}

PyScatteredInterp::CLS PyScatteredInterp::init(std::nullptr_t, const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first) {
    return {tri.tri(), values, coord_axis_first};
}

int PyScatteredInterp::ndim(const CLS& self) {
    return self.ndim();
}

py::object PyScatteredInterp::py_points(const CLS& self){
    return py::cast(self.points());
}

py::object PyScatteredInterp::py_values(const CLS& self){
    return py::cast(self.values());
}

py::object PyScatteredInterp::py_delaunay_obj(const CLS& self){
    return py::cast(PyDelaunay(self.tri()));
    //TODO: do nat cast to pointer, return actual object, but in a memory-cheap way.
}

std::nullptr_t PyScatteredInterp::parse_args(const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first){
    if (x.ndim() > 2){
        throw py::value_error("ScatteredNdInterpolator requires a 2D array for the input points (or optionally a 1D array for 1D fields)");
    }else if ((values.ndim() == 1 || coord_axis_first) && values.shape(0) != x.shape(0)){
        throw py::value_error("The shape of values along axis 0 must be equal to the number of points");
    }else if (values.ndim() > 1 && !coord_axis_first && values.shape(values.ndim()-1) != x.shape(0)){
        throw py::value_error("The shape of values along axis " + std::to_string(values.ndim()-1) + " must be equal to the number of points");
    }else if (x.ndim() == 2 && x.shape(0) < x.shape(1)+1){
        throw py::value_error("Number of input points must be at least one more than the number of dimensions");
    }
    return nullptr;
}

std::nullptr_t PyScatteredInterp::parse_tri_args(const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first){
    if (values.ndim() == 0){
        throw py::value_error("Values must be at least a 1D array, not scalar");
    }else if(values.ndim() > 1 && !coord_axis_first && tri.py_npoints() != int(values.shape(values.ndim()-1))){
        throw py::value_error("Number of field values along axis " + std::to_string(values.ndim()-1) + " must match number of points in the Delaunay triangulation");
    }else if(values.ndim() > 1 && coord_axis_first && tri.py_npoints() != int(values.shape(0))){
        throw py::value_error("Number of field values along axis 0 must match number of points in the Delaunay triangulation");
    }
    return nullptr;
}


} // namespace ode
