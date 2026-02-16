#ifndef PY_FIELD_IMPL_HPP
#define PY_FIELD_IMPL_HPP

#include "../../../include/pyode/lib/PyOde.hpp"
#include "../../../include/pyode/lib/PyResult.hpp"
#include "../../../include/pyode/lib/PyField.hpp"
#include "../../../include/ode/Interpolation/VectorFields_impl.hpp"
#include "../../../include/ode/Interpolation/NdInterpolator_impl.hpp"
#include "../../../include/ode/Interpolation/Regular/RegularGridInterpolator_impl.hpp"
#include "../../../include/ode/Interpolation/Scattered/ScatteredNdInterpolator_impl.hpp"
#include "../../../include/pyode/pycast/pycast.hpp"



namespace ode{

//Explicit instanciation

template class RegularVectorField<double, 0, true>;
template class ScatteredVectorField<0, true>;

// ============================================================================================
//                                      PyScalarField
// ============================================================================================

py::object PyScalarField::py_value_at(const VirtualNdInterpolator& self, const py::args& x){
    return PyNdInterp::py_value_at(self, x);
}

// ============================================================================================
//                              PyRegScalarField
// ============================================================================================

py::array_t<double> PyRegScalarField::parse_values(const py::array_t<double>& values, const py::args& py_grid){
    if (size_t(values.ndim()) != py_grid.size()){
        throw py::value_error("Values array must have the same number of dimensions as the number of grid arrays");
    }
    return values;
}

PyRegScalarField::PyRegScalarField(const py::array_t<double>& values, const py::args& py_grid) : RGBase(PyRegGridInterp::init_main(parse_values(values, py_grid), py_grid)) {}

// ============================================================================================
//                                      PyScatteredField
// ============================================================================================

PyScatteredField::PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values) : PyScatteredField(parse_values(values), x, values) {}

PyScatteredField::PyScatteredField(const PyDelaunay& tri, const py::array_t<double>& values) : PyScatteredField(parse_values(values), tri.tri(), values) {}

py::object PyScatteredField::py_delaunay() const { return py::cast(PyDelaunay(this->tri())); }

py::object PyScatteredField::py_points() const { return py::cast(this->points()); }

py::object PyScatteredField::py_values() const { return py::cast(this->values()); }

std::nullptr_t PyScatteredField::parse_values(const py::array_t<double>& values){
    if (values.ndim() != 1){
        throw py::value_error("Values array must be 1D for ScatteredScalarField");
    }
    return nullptr;
}

PyScatteredField::PyScatteredField(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values) : InterpBase(PyScatteredInterp::init_main(x, values)) {}

PyScatteredField::PyScatteredField(std::nullptr_t, const PyDelaunay& tri, const py::array_t<double>& values) : InterpBase(PyScatteredInterp::init_tri(tri, values)) {}

// ============================================================================================
//                              PyVecField
// ============================================================================================


void PyVecField::check_coords(const CLS& self, const double* coords){
    if (!self.contains(coords)){
        Array1D<double, 0> q(coords, self.ndim());
        throw py::value_error("Coordinates " + py::repr(py::cast(q)).template cast<std::string>() + " are out of bounds");
    }
}

py::object PyVecField::py_streamline(const CLS& self, const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method, bool normalized){

    if (q0.ndim() != 1 || q0.shape(0) != self.ndim()){
        throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(self.ndim()));
    }

    check_coords(self, q0.data());

    StepSequence<double> t_seq = to_step_sequence<double>(t_eval);
    try{
        double max_step_val = (max_step.is_none() ? inf<double>() : max_step.cast<double>());
        auto* result = new OdeResult<double>(self.streamline(q0.data(), length, rtol, atol, min_step, max_step_val, stepsize, direction, t_seq, method.cast<std::string>(), normalized));
        PyOdeResult py_res(result, {self.ndim()}, DTYPE_MAP.at("double"));
        return py::cast(py_res);
    } catch (const std::runtime_error& e){
        throw py::value_error(e.what());
    }
}

py::object PyVecField::py_streamline_ode(const CLS& self, const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized){
    if (direction != 1 && direction != -1){
        throw py::value_error("Direction must be either 1 (forward) or -1 (backward)");
    }else if (q0.ndim() != 1 || q0.shape(0) != self.ndim()){
        throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(self.ndim()));
    }

    check_coords(self, q0.data());
    ODE<double>* ode = self.get_streamline_ode(q0.data(), rtol, atol, min_step, max_step.is_none() ? inf<double>() : max_step.cast<double>(), stepsize, direction, method.cast<std::string>(), normalized);

    return py::cast(PyODE(ode, get_scalar_type<double>()));

}


// ============================================================================================
//                              PyRegVecField
// ============================================================================================

void PyRegVecField::parse_values(const py::array_t<double>& values, const py::args& py_grid){
    if (size_t(values.shape(0)) != py_grid.size()){
        throw py::value_error("Size of values along axis 0 (number of vector components) of the vector field must be equal to the number of grid arrays");
    }else if (py_grid.size() < 2){
        throw py::value_error("At least 2 components are required for a vector field");
    }
}


PyRegVecField::CLS PyRegVecField::init_main(const py::array_t<double>& values, const py::args& py_grid){
    parse_values(values, py_grid);
    auto grid = RGBase::parse_args(values, py_grid, false);
    const double* v_data = values.data();
    // values shape right not is (ndim, ...), where (...) has a product of n_points.
    // We need to reshape to (ndim, n_points)
    View2D<double> v_view(v_data, values.shape(0), values.size() / values.shape(0));
    return {v_view, grid, false};
}


py::object PyRegVecField::py_streamplot_data(const CLS& self, double max_length, double ds, int density){
    if (density <= 1){
        throw py::value_error("Density must be greater than 1");
    }
    if (max_length <= 0){
        throw py::value_error("Max length must be a positive number");
    }
    if (ds <= 0){
        throw py::value_error("ds must be a positive number");
    }

    std::vector<Array2D<double, 0, 0>> streamlines = self.streamplot_data(max_length, ds, size_t(density));
    py::list result;
    for (const Array2D<double, 0, 0>& line : streamlines){
        result.append(py::cast(line));
    }
    return result;
}


PyScatVecField::CLS PyScatVecField::init(const py::array_t<double>& x, const py::array_t<double>& values){
    parse_values(values);
    SCBase::parse_args(x, values, false);
    const int ndim = (x.ndim() == 1) ? 1 : int(x.shape(1));
    const double* x_data = x.data();
    return {x_data, values, ndim, false};
}


PyScatVecField::CLS PyScatVecField::init_tri(const PyDelaunay& tri, const py::array_t<double>& values){
    parse_values(values);
    SCBase::parse_tri_args(tri, values, false);
    return {tri.tri(), values, false};
}

void PyScatVecField::parse_values( const py::array_t<double>& values){
    if (values.ndim() != 2){
        throw py::value_error("Values array must be 2D for ScatteredVectorField");
    }
}

} // namespace ode

#endif // PY_FIELD_IMPL_HPP