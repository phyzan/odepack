#ifndef LINEAR_ND_INTERPOLATOR_HPP
#define LINEAR_ND_INTERPOLATOR_HPP

#include "../../ndspan.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>


using namespace ndspan;

namespace ode{

template<size_t NDIM>
class DelaunayTri{

    static constexpr double EPS = std::numeric_limits<double>::epsilon() * 100;

    static constexpr Allocation SPX_ALLOC = NDIM == 0 ? Allocation::Heap : Allocation::Stack;

public:

    using PointStorage = Array2D<double, 0, NDIM>;
    static constexpr int DIM_SPX = (NDIM == 0 ? 0 : NDIM+1);

    DEFAULT_RULE_OF_FOUR(DelaunayTri)

    template<typename PointsArrayType>
    DelaunayTri(const PointsArrayType& points);

    int                 ndim() const;
    int                 npoints() const;
    const PointStorage& get_points() const;
    int                 nsimplices() const;
    const Array2D<int, 0, DIM_SPX>& get_simplices() const;
    const int*          get_simplex(int simplex_idx) const; //ndim + 1 elements
    const int*          get_neighbors(int simplex_idx) const; // ndim + 1 elements, -1 = no neighbor (boundary)
    int                 find_simplex(const double* point) const; // array of ndim elements, returns simplex index or -1 if not found
    bool                compute_barycentric(double* out, int simplex_idx, const double* point) const; // bary should have ndim+1 elements, returns false if simplex is degenerate
    

    // if n_fields > 1, then field must be a contiguous array of shape (n_fields, npoints), and output of interpolate will be a contiguous array of shape (n_fields,)
    void                interpolate(double* out, const double* point, const double* field, int n_fields=1) const;

    double              get_value(const double* point, const double* field) const; // field should have npoints elements, returns interpolated value at point

    // auxiliary function for interpolation, used when the caller already has the barycentric coordinates and simplex index (e.g. for multiple fields)
    double              weighted_field(const double* field, const double* bary, const int* simplex) const; // res = bary[i] * field[simplex[i]], i=0..ndim (ndim+1 terms)

private:

    void compute_delaunay_1d();

    int& thread_cache() const;


    // ============================================================================
    // Qhull-based Delaunay triangulation - O(n log n)
    // ============================================================================

    void compute_delaunay_nd();

    void compute_delaunay();

    PointStorage                    points_;    // (npoints, ndim)
    Array2D<int, 0, DIM_SPX>        simplices_; // (nsimplices, ndim+1)
    Array2D<int, 0, DIM_SPX>        neighbors_; // (nsimplices, ndim+1), -1 = boundary
    Array2D<double, 0, NDIM>        v0_;        // (nsimplices, ndim)
    Array3D<double, 0, NDIM, NDIM>  invT_;      // (nsimplices, ndim, ndim)

};


template<size_t NDIM>
class LinearNdInterpolator : public DelaunayTri<NDIM>{

    using Base = DelaunayTri<NDIM>;

public:

    template<typename PointsArrayType, typename FieldContainer>
    LinearNdInterpolator(const PointsArrayType& points, const FieldContainer& fields);

    template<typename FieldContainer>
    LinearNdInterpolator(const DelaunayTri<NDIM>& tri, const FieldContainer& fields);

    int nfields() const;

    View1D<double> get_field(int field_idx = 0) const;

    double interp_single(const double* query_point, int field_idx = 0) const;

    void interpolate(double* out, const double* point) const;

private:

    Array2D<double, 0, 0> values_;

};


template<size_t NDIM>
class ScatteredScalarField : private LinearNdInterpolator<NDIM>{

    using Base = LinearNdInterpolator<NDIM>;

public:

    // Expose needed methods from DelaunayTri base class
    using DelaunayTri<NDIM>::ndim;
    using DelaunayTri<NDIM>::npoints;
    using DelaunayTri<NDIM>::get_points;

    ScatteredScalarField(const DelaunayTri<NDIM>& tri, const double* values);

    template<typename PointsArrayType>
    ScatteredScalarField(const PointsArrayType& points, const double* values);

    const DelaunayTri<NDIM>& get_delaunay() const;

    const double* get_field() const;

    double get_value(const double* query_point) const;

    template<typename... Scalar>
    double operator()(Scalar... coords) const;

};






} // namespace ode

#endif // LINEAR_ND_INTERPOLATOR_HPP
