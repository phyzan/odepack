#ifndef PY_INTERP_HPP
#define PY_INTERP_HPP


#include "../../ode/Interpolation/Regular/RegularGridInterpolator.hpp"
#include "../../ode/Interpolation/Scattered/ScatteredNdInterpolator.hpp"
#include "PyTools.hpp"

namespace ode{


struct PyNdInterp {

    static py::object py_value_at(const VirtualNdInterpolator& self, const py::args& x);

    static int ndim(const VirtualNdInterpolator& self);

}; // struct PyNdInterp



struct PyRegGridInterp {

    using CLS = RegularGridInterpolator<double, 0, true>;

    // ======================= Python interface =====================================
    static CLS init_main(const py::array_t<double>& values, const py::args& py_grid);

    static py::object get_values(const CLS& self);

    static py::tuple get_grid(const CLS& self);

    // ==============================================================================

    static CLS init(const py::array_t<double>& values, const std::vector<Array1D<double>>& grid, bool coord_axis_first);

    static CLS init(const py::array_t<double>& values, const py::args& py_grid, bool coord_axis_first);

    static std::vector<Array1D<double>> parse_args(const py::array_t<double>& values, const py::args& py_grid, bool coord_axis_first);

}; // class PyRegGridInterp



class PyDelaunay {

    using Base = DelaunayTri<0>;

public:

    PyDelaunay(const py::array_t<double>& x);

    PyDelaunay(TriPtr<0> tri);

    py::object py_points() const;

    int py_ndim() const;
    int py_npoints() const;
    int py_nsimplices() const;
    int py_find_simplex(const py::array_t<double>& point) const;

    //returns array of shape (ndim+1,), or None if point is out of bounds.
    py::object py_get_simplex(const py::array_t<double>& point) const;
    
    // array of shape (nsimplices, ndim+1) containing the indices of the points that form each simplex
    py::object py_get_simplices() const;

    TriPtr<0> tri() const;

    // ================ Pickling ===================
    py::dict    py_get_state() const;
    
    static PyDelaunay py_set_state(const py::dict& state);
    // =============================================
    

private:

    PyDelaunay(std::nullptr_t, const py::array_t<double>& x);

    PyDelaunay() = default; //for pickling, not called from outside

    static std::nullptr_t parse_args(const py::array_t<double>& x);

    TriPtr<0> tri_;

}; // class PyDelaunay


struct PyScatteredInterp {

    using CLS = ScatteredNdInterpolator<0, true>;

    // Python signature is ScatteredNdInterpolator(x: np.ndarray (npoints, ndim), values: np.ndarray (npoints, ...)), where
    static CLS init_main(const py::array_t<double>& x, const py::array_t<double>& values);

    static CLS init_tri(const PyDelaunay& tri, const py::array_t<double>& values);

    static CLS init(const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first);

    static CLS init(const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first);

    static py::object py_delaunay_obj(const CLS& self);

    static int ndim(const CLS& self);

    static py::object py_points(const CLS& self);

    static py::object py_values(const CLS& self);

    // Internal interface

    static CLS init(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first);

    static CLS init(std::nullptr_t, const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first);

    static std::nullptr_t parse_args(const py::array_t<double>& x, const py::array_t<double>& values, bool coord_axis_first);
    static std::nullptr_t parse_tri_args(const PyDelaunay& tri, const py::array_t<double>& values, bool coord_axis_first);


}; // struct PyScatteredInterp


} // namespace ode


#endif // PY_INTERP_HPP