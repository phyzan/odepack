#ifndef PY_FIELD_HPP
#define PY_FIELD_HPP

#include "../../ode/Interpolation/SampledVectorfields_impl.hpp"
#include "../../ode/Interpolation/LinearNdInterpolator.hpp"
#include "../pytools/pytools.hpp"

namespace ode{

template<size_t NDIM>
class PyScalarField : public RegularGridInterpolator<double, NDIM> {

    using Base = RegularGridInterpolator<double, NDIM>;

private:

    template<size_t... I, typename... PyArray>
    PyScalarField(std::nullptr_t, std::index_sequence<I...>, const PyArray&... args);

public:

    template<typename... PyArray>
    PyScalarField(const PyArray&... args);

    template<typename... Scalar>
    double operator()(Scalar... x) const;

private:

    template<typename... PyArray>
    static std::nullptr_t parse_args(const PyArray&... args);
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
    

private:

    PyDelaunay(std::nullptr_t, const py::array_t<double>& x);

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


class PyVecFieldBase {

public:

    virtual ~PyVecFieldBase() = default;

    virtual py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const = 0;

    virtual py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const = 0;

    virtual py::object py_streamplot_data(double max_length, double ds, int density) const = 0;

};


template<size_t NDIM>
class PyVecField : public SampledVectorField<double, NDIM>, public PyVecFieldBase {

    using Base = SampledVectorField<double, NDIM>;

private:

    template<size_t... I, typename... PyArray>
    PyVecField(std::nullptr_t, std::index_sequence<I...>, const PyArray&... args);

public:

    template<typename... PyArray>
    PyVecField(const PyArray&... args);

    template<size_t Axis>
    py::object py_x() const;

    template<size_t FieldIdx>
    py::object py_vx() const;

    template<size_t FieldIdx, typename... Scalar>
    py::object py_vx_at(Scalar... x) const;

    template<typename... Scalar>
    py::object py_vector(Scalar... x) const;

    template<typename... Scalar>
    bool py_in_bounds(Scalar... x) const;

    // ==========================================

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const override;

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const override;

    py::object py_streamplot_data(double max_length, double ds, int density) const override;

private:

    template<typename... Scalar>
    void check_coords(Scalar... x) const;

    template<typename... PyArray>
    static std::nullptr_t parse_args(const PyArray&... args);



};


class PyVecField2D : public PyVecField<2> {
    
    using Base = PyVecField<2>;

public:

    PyVecField2D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& vx, const py::array_t<double>& vy);

    using Base::py_streamline, Base::py_streamline_ode, Base::py_streamplot_data;

};

class PyVecField3D : public PyVecField<3> {
    
    using Base = PyVecField<3>;

public:

    PyVecField3D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& z, const py::array_t<double>& vx, const py::array_t<double>& vy, const py::array_t<double>& vz);

    using Base::py_streamline, Base::py_streamline_ode, Base::py_streamplot_data;
    
};


}


#endif // PY_FIELD_HPP