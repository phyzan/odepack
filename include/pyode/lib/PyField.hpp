#ifndef PY_FIELD_HPP
#define PY_FIELD_HPP

#include "../../ode/Interpolation/SampledVectorfields.hpp"
#include "../../ode/Interpolation/LinearNdInterpolator.hpp"
#include "PyTools.hpp"

namespace ode{


class PyScalarField : public RegularGridInterpolator<double, 0> {

    using Base = RegularGridInterpolator<double, 0>;

private:


    PyScalarField(const double* values, const std::vector<MutView1D<double>>& grid);

public:

    PyScalarField(const py::array_t<double>& values, const py::args& py_grid);

    template<typename... Scalar>
    double operator()(Scalar... x) const;

    double py_value_at(const py::args& x) const;

protected:
    PyScalarField(const py::array_t<double>& values, const py::args& py_grid, size_t NDIM);
    
private:

    static std::vector<MutView1D<double>> parse_args(const py::array_t<double>& values, const py::args& py_grid, size_t NDIM);
};

class PyScalarField1D : public PyScalarField {

    using Base = PyScalarField;
public:
    
    PyScalarField1D(const py::array_t<double>& values, const py::args& py_grid);
    double operator()(double x) const;

};

class PyScalarField2D : public PyScalarField {

    using Base = PyScalarField;
public:

    PyScalarField2D(const py::array_t<double>& values, const py::args& py_grid);
    double operator()(double x, double y) const;

};

class PyScalarField3D : public PyScalarField {

    using Base = PyScalarField;
public:
    PyScalarField3D(const py::array_t<double>& values, const py::args& py_grid);
    double operator()(double x, double y, double z) const;

};


class PyDelaunayBase{

public:

    virtual ~PyDelaunayBase() = default;
    virtual py::object py_points() const = 0;
    virtual int py_ndim() const = 0;
    virtual int py_npoints() const = 0;
    virtual int py_nsimplices() const = 0;
    virtual int py_find_simplex(const py::array_t<double>& point) const = 0;
    virtual py::object py_get_simplices() const = 0;
    virtual py::object py_get_simplex(const py::array_t<double>& point) const = 0;
    virtual py::dict py_get_state() const = 0;
};


template<size_t NDIM>
class PyDelaunay : public DelaunayTri<NDIM>, public PyDelaunayBase {

    using Base = DelaunayTri<NDIM>;

public:

    PyDelaunay(const py::array_t<double>& x);

    py::object py_points() const override;

    int py_ndim() const override;
    int py_npoints() const override;
    int py_nsimplices() const override;
    int py_find_simplex(const py::array_t<double>& point) const override;

    //returns array of shape (ndim+1,), or None if point is out of bounds.
    py::object py_get_simplex(const py::array_t<double>& point) const override;
    
    // array of shape (nsimplices, ndim+1) containing the indices of the points that form each simplex
    py::object py_get_simplices() const override;

    // ================ Pickling ===================
    py::dict    py_get_state() const override;
    
    static PyDelaunay py_set_state(const py::dict& state);
    
    // =============================================
    

private:

    PyDelaunay(std::nullptr_t, const py::array_t<double>& x);

    PyDelaunay() = default; //for pickling, not called from outside

    static std::nullptr_t parse_args(const py::array_t<double>& x);

};


template<size_t NDIM>
class PyScatteredField : public ScatteredScalarField<NDIM> {

    using Base = ScatteredScalarField<NDIM>;

public:

    // Python signature is ScatteredField(x: np.ndarray (npoints, ndim), values: np.ndarray (npoints)), where
    PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values);

    PyScatteredField(const PyDelaunay<NDIM>& tri, const py::array_t<double>& values);

    template<typename... Scalar>
    double operator()(Scalar... x) const;

    py::object py_points() const;

    py::object py_values() const;

    py::object py_delaunay_obj() const;

    double py_value_at(const py::args& x) const;

private:

    PyScatteredField(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values);

    PyScatteredField(std::nullptr_t, const PyDelaunay<NDIM>& tri, const py::array_t<double>& values);

    static std::nullptr_t parse_args(const py::array_t<double>& x, const py::array_t<double>& values);

    static std::nullptr_t parse_tri_args(const PyDelaunay<NDIM>& tri, const py::array_t<double>& values);

};


template<size_t NDIM>
class PyVecField : public SampledVectorField<double, NDIM> {

    using Base = SampledVectorField<double, NDIM>;

private:

    PyVecField(const double* values, const std::vector<MutView1D<double>>& grid);

public:

    PyVecField(const py::array_t<double>& values, const py::args& py_grid);

    py::object py_x(int axis) const;

    py::object py_vx(int component) const;

    py::object py_coords() const;

    py::object py_data() const;

    py::object py_value(const py::args& coords) const;

    bool py_in_bounds(const py::args& coords) const;

    // ==========================================

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const;

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const;

    py::object py_streamplot_data(double max_length, double ds, int density) const;

private:

    void check_coords(const double* coords) const;

    static std::vector<MutView1D<double>> parse_args(const py::array_t<double>& values, const py::args& py_grid);

};

// Inheriting from <0>, not <2> for simpler compilation and inheritance chain
class PyVecField2D : public PyVecField<0> {

    using Base = PyVecField<0>;

public:    

    using Base::Base;

    py::object x() const;
    py::object y() const;
    py::object vx() const;
    py::object vy() const;

};


class PyVecField3D : public PyVecField<0> {

    using Base = PyVecField<0>;

public:

    using Base::Base;

    py::object x() const;
    py::object y() const;
    py::object z() const;
    py::object vx() const;
    py::object vy() const;
    py::object vz() const;

};

} // namespace ode

#endif // PY_FIELD_HPP