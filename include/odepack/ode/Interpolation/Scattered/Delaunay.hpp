#ifndef DELAUNAY_HPP
#define DELAUNAY_HPP

#include <memory>
#include "../../../ndspan.hpp"

namespace ode {

using namespace ndspan;

template<size_t NDIM>
class DelaunayTri{

    static constexpr double EPS = std::numeric_limits<double>::epsilon() * 100;

    static constexpr Allocation SPX_ALLOC = NDIM == 0 ? Allocation::Heap : Allocation::Stack;

public:

    static constexpr int DIM_SPX = (NDIM == 0 ? 0 : NDIM+1);

    DelaunayTri(const DelaunayTri& other) = delete;

    DelaunayTri& operator=(const DelaunayTri& other) = delete;

    DelaunayTri(DelaunayTri&& other) noexcept = default;

    DelaunayTri& operator=(DelaunayTri&& other) noexcept = default;

    DelaunayTri(const double* points, size_t n_points, size_t ndim);

    // Allowes derived classes to set a state into the object
    // wihtout having to recompute the triangulation. Used for pickling in Python wrapper.
    DelaunayTri() = default;

    void set_state(const View2D<double, 0, NDIM>& points, const View2D<int, 0, DIM_SPX>& simplices, const View2D<int, 0, DIM_SPX>& neighbors, const View2D<double, 0, NDIM>& v0, const View3D<double, 0, NDIM, NDIM>& invT);

    // Member variable accessors
    const Array2D<double, 0, NDIM>& get_points() const; // (n_points, ndim)
    const Array2D<int, 0, DIM_SPX>& get_simplices() const; // (nsimplices, ndim+1)
    const Array2D<int, 0, DIM_SPX>& get_neighbors() const; // (nsimplices, ndim+1)
    const Array2D<double, 0, NDIM>& get_v0() const; // (nsimplices, ndim)
    const Array3D<double, 0, NDIM, NDIM>& get_invT() const; // (nsimplices, ndim, ndim)
    

    int                 ndim() const;
    int                 npoints() const;
    
    int                 nsimplices() const;
    
    const int*          get_simplex(int simplex_idx) const; //ndim + 1 elements
    const int*          get_neighbors(int simplex_idx) const; // ndim + 1 elements, -1 = no neighbor (boundary)
    int                 find_simplex(const double* point) const; // array of ndim elements, returns simplex index or -1 if not found
    bool                contains(const double* point) const { return find_simplex(point) != -1; }
    bool                compute_barycentric(double* out, int simplex_idx, const double* point) const; // bary should have ndim+1 elements, returns false if simplex is degenerate

    // if n_fields > 1, then field must be a contiguous array of shape (n_fields, npoints), and output of interpolate will be a contiguous array of shape (n_fields,)
    bool                interpolate(double* out, const double* point, const double* field, int n_fields=1) const;

    double              get_value(const double* point, const double* field) const; // field should have npoints elements, returns interpolated value at point

    double              volume(int simplex_idx) const; // returns the volume of the simplex with index simplex_idx

    double              total_volume() const;

    // auxiliary function for interpolation, used when the caller already has the barycentric coordinates and simplex index (e.g. for multiple fields)


private:

    void compute_delaunay_1d();

    int& thread_cache() const;

    Array2D<double, NDIM, NDIM>& mat_cache() const;


    // ============================================================================
    // Qhull-based Delaunay triangulation - O(n log n)
    // ============================================================================

    void compute_delaunay_nd();

    void compute_delaunay();

    Array2D<double, 0, NDIM>        points_;    // (npoints, ndim)
    Array2D<int, 0, DIM_SPX>        simplices_; // (nsimplices, ndim+1)
    Array2D<int, 0, DIM_SPX>        neighbors_; // (nsimplices, ndim+1), -1 = boundary
    Array2D<double, 0, NDIM>        v0_;        // (nsimplices, ndim)
    Array3D<double, 0, NDIM, NDIM>  invT_;      // (nsimplices, ndim, ndim)

};

template<size_t NDIM>
using TriPtr = std::shared_ptr<DelaunayTri<NDIM>>;

} // namespace ode


#endif // DELAUNAY_HPP