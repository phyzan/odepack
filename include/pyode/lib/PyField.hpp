#ifndef PY_FIELD_HPP
#define PY_FIELD_HPP

#include "../../ode/Interpolation/VectorFields.hpp"
#include "PyInterp.hpp"
#include "PyOde.hpp"
#include "PyResult.hpp"

namespace ode{


struct PyScalarField {

    // returns double (not array), as this is a scalar field.
    static py::object py_value_at(const VirtualNdInterpolator& self, const py::args& x);
}; // struct PyScalarField



class PyRegScalarField : public RegularGridInterpolator<double, 0, true>, public PyScalarField {

public:

    using RGBase = RegularGridInterpolator<double, 0, true>;

    static py::array_t<double> parse_values(const py::array_t<double>& values, const py::args& py_grid);
    
    PyRegScalarField(const py::array_t<double>& values, const py::args& py_grid);
    

    // Keep in the header file for easier instantiation from dynamically compiled ODE code
    // regardless of the number of dimensions, without needing to explicitly instantiate for each dimension in the precompiled library.
    template<typename... Scalar>
    double operator()(Scalar... x) const{
        assert(sizeof...(Scalar) == this->ndim() && "Number of coordinates must match number of dimensions of the field");
        double coords[] = {double(x)...};
        double out;
        if (!this->interp(&out, coords)){
            return std::numeric_limits<double>::quiet_NaN();
        }else {
            return out;
        }
    }

}; // class PyRegScalarField



class PyScatteredField : public ScatteredNdInterpolator<0, true>, public PyScalarField {


public:

    using InterpBase = ScatteredNdInterpolator<0, true>;

    // Python signature is ScatteredField(x: np.ndarray (npoints, ndim), values: np.ndarray (npoints,)
    PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values);

    PyScatteredField(const PyDelaunay& tri, const py::array_t<double>& values);

    py::object py_delaunay() const;

    py::object py_points() const;

    py::object py_values() const;

    static std::nullptr_t parse_values( const py::array_t<double>& values);

    // Keep in the header file for easier instantiation from dynamically compiled ODE code
    // regardless of the number of dimensions, without needing to explicitly instantiate for each dimension in the precompiled library.
    template<typename... Scalar>
    double operator()(Scalar... x) const{
        double coords[] = {double(x)...};
        assert(sizeof...(Scalar) == this->ndim() && "Number of coordinates must match number of dimensions of the field");
        double out;
        if (!this->interp(&out, coords)){
            return std::numeric_limits<double>::quiet_NaN();
        }else {
            return out;
        }
    }

private:

    PyScatteredField(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values);

    PyScatteredField(std::nullptr_t, const PyDelaunay& tri, const py::array_t<double>& values);

}; // class PyScatteredField




struct PyVecField {

    // returns array, as this is a vector field.

    using CLS = VirtualVectorField;

    static void check_coords(const CLS& self, const double* coords);

    static py::object py_streamline(const CLS& self, const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method, bool normalized);

    static py::object py_streamline_ode(const CLS& self, const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized);
    
}; // class PyVecField


struct PyRegVecField {

    using RGBase = PyRegGridInterp;
    using VFBase = PyVecField;

    using CLS = RegularVectorField<double, 0, true>;

    // ============================= Python interface =============================
    
    static CLS init_main(const py::array_t<double>& values, const py::args& py_grid);

    static py::object py_streamplot_data(const CLS& self, double max_length, int density, double ds, double rtol, double atol, double min_step, const py::object& max_step, const py::str& method);

    static py::object component(const CLS& self, int i);

    // ============================================================================

    static void parse_values(const py::array_t<double>& values, const py::args& py_grid);

}; // class PyRegVecField


struct PyScatVecField {

    using SCBase = PyScatteredInterp;
    using VFBase = PyVecField;

    using CLS = ScatteredVectorField<0, true>;


public:

    static CLS init(const py::array_t<double>& x, const py::array_t<double>& values);

    static CLS init_tri(const PyDelaunay& tri, const py::array_t<double>& values);

    static void parse_values( const py::array_t<double>& values);

}; // class PyScatVecField

} // namespace ode

#endif // PY_FIELD_HPP