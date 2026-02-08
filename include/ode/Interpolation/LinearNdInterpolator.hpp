#ifndef LINEAR_ND_INTERPOLATOR_HPP
#define LINEAR_ND_INTERPOLATOR_HPP

#include "../../ndspan.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace ndspan;

template<typename T>
class LinearNdInterpolator {
    // Linear N-dimensional interpolator for scattered data points.
    // Uses Delaunay triangulation to partition the point set into simplices,
    // then performs barycentric interpolation within the containing simplex.
    //
    // The class offers the option to store many fields at once, on the same set of points,
    // for efficient interpolation of vector fields. The constructor takes as input a set of
    // N-dimensional points and corresponding field values, and the interpolation method
    // computes the interpolated field value at any query point in the convex hull of the input points.

public:

    template<typename PointsArrayType, typename FieldContainer>
    LinearNdInterpolator(const PointsArrayType& points, const FieldContainer& fields)
        : points_(points.data(), points.shape(0), (points.ndim() == 1 ? 1 : points.shape(1))),
          values_(fields.size(), points.shape(0)),
          ndim_(points.ndim() == 1 ? 1 : points.shape(1)) {

        assert(points.ndim() < 3 && "Points array must be 2D (shape = (Npoints, Ndim))");
        assert(fields.size() > 0 && "At least one field must be provided");

        // Copy field data into a contiguous array
        size_t Npoints = points.shape(0);
        for (size_t field_idx = 0; field_idx < fields.size(); field_idx++) {
            const T* field_data = fields[field_idx];
            copy_array(values_.data() + field_idx * Npoints, field_data, Npoints);
        }

        // Compute Delaunay triangulation
        compute_delaunay();
    }

    inline size_t ndim() const{
        return points_.Ncols();
    }

    inline size_t npoints() const{
        return points_.Nrows();
    }

    inline const Array2D<T, 0, 0>& get_points() const{
        return points_;
    }

    inline size_t nfields() const{
        return values_.Nrows();
    }

    inline View1D<T> get_field(size_t field_idx = 0) const{
        assert(field_idx < nfields() && "Field index out of bounds");
        const T* field_data = values_.ptr(field_idx, 0);
        return View1D<T>(field_data, npoints());
    }

    inline size_t nsimplices() const {
        return simplices_.size() / (ndim_ + 1);
    }

    // Get vertex indices for a simplex (returns pointer to ndim+1 indices)
    inline const size_t* get_simplex(size_t simplex_idx) const {
        assert(simplex_idx < nsimplices() && "Simplex index out of bounds");
        return simplices_.data() + simplex_idx * (ndim_ + 1);
    }

    // Find the simplex containing the query point
    // Returns -1 if point is outside the convex hull
    int find_simplex(const T* query_point) const {
        size_t num_simplices = nsimplices();
        std::vector<T> bary(ndim_ + 1);

        for (size_t s = 0; s < num_simplices; ++s) {
            if (compute_barycentric(s, query_point, bary.data())) {
                // Check if all barycentric coordinates are non-negative (with tolerance)
                bool inside = true;
                for (size_t i = 0; i <= ndim_; ++i) {
                    if (bary[i] < -eps_) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    return static_cast<int>(s);
                }
            }
        }
        return -1;
    }

    // Interpolate a single field at the query point
    // Returns NaN if point is outside the convex hull
    T operator()(const T* query_point, size_t field_idx = 0) const {
        assert(field_idx < nfields() && "Field index out of bounds");

        int simplex_idx = find_simplex(query_point);
        if (simplex_idx < 0) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        std::vector<T> bary(ndim_ + 1);
        compute_barycentric(static_cast<size_t>(simplex_idx), query_point, bary.data());

        const size_t* vertices = get_simplex(static_cast<size_t>(simplex_idx));
        T result = T(0);
        for (size_t i = 0; i <= ndim_; ++i) {
            result += bary[i] * values_(field_idx, vertices[i]);
        }
        return result;
    }

    // Interpolate all fields at the query point
    // Output array must have size >= nfields()
    void interpolate(const T* query_point, T* output) const {
        int simplex_idx = find_simplex(query_point);
        if (simplex_idx < 0) {
            for (size_t f = 0; f < nfields(); ++f) {
                output[f] = std::numeric_limits<T>::quiet_NaN();
            }
            return;
        }

        std::vector<T> bary(ndim_ + 1);
        compute_barycentric(static_cast<size_t>(simplex_idx), query_point, bary.data());

        const size_t* vertices = get_simplex(static_cast<size_t>(simplex_idx));
        for (size_t f = 0; f < nfields(); ++f) {
            T result = T(0);
            for (size_t i = 0; i <= ndim_; ++i) {
                result += bary[i] * values_(f, vertices[i]);
            }
            output[f] = result;
        }
    }

private:
    static constexpr T eps_ = std::numeric_limits<T>::epsilon() * T(100);

    Array2D<T, 0, 0> points_; // shape = (POINTS, NDIM)
    Array2D<T, 0, 0> values_; // shape = (FIELDS, POINTS)
    size_t ndim_;
    std::vector<size_t> simplices_; // Flattened simplex vertex indices, each simplex has (ndim+1) vertices

    // Compute barycentric coordinates of a point within a simplex
    // Returns false if the computation fails (degenerate simplex)
    bool compute_barycentric(size_t simplex_idx, const T* point, T* bary) const {
        const size_t* vertices = get_simplex(simplex_idx);
        size_t n = ndim_;

        // Build matrix T where T[i][j] = points_[vertices[j]][i] - points_[vertices[n]][i]
        // This is the transformation matrix from barycentric to Cartesian (minus last vertex)
        std::vector<T> mat(n * n);
        std::vector<T> rhs(n);

        for (size_t i = 0; i < n; ++i) {
            rhs[i] = point[i] - points_(vertices[n], i);
            for (size_t j = 0; j < n; ++j) {
                mat[i * n + j] = points_(vertices[j], i) - points_(vertices[n], i);
            }
        }

        // Solve mat * lambda = rhs using Gaussian elimination with partial pivoting
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        for (size_t k = 0; k < n; ++k) {
            // Find pivot
            T max_val = std::abs(mat[perm[k] * n + k]);
            size_t max_idx = k;
            for (size_t i = k + 1; i < n; ++i) {
                T val = std::abs(mat[perm[i] * n + k]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            if (max_val < eps_) {
                return false; // Singular matrix (degenerate simplex)
            }

            std::swap(perm[k], perm[max_idx]);

            // Eliminate
            for (size_t i = k + 1; i < n; ++i) {
                T factor = mat[perm[i] * n + k] / mat[perm[k] * n + k];
                for (size_t j = k + 1; j < n; ++j) {
                    mat[perm[i] * n + j] -= factor * mat[perm[k] * n + j];
                }
                rhs[perm[i]] -= factor * rhs[perm[k]];
            }
        }

        // Back substitution
        for (size_t k = n; k-- > 0;) {
            T sum = rhs[perm[k]];
            for (size_t j = k + 1; j < n; ++j) {
                sum -= mat[perm[k] * n + j] * bary[j];
            }
            bary[k] = sum / mat[perm[k] * n + k];
        }

        // Last barycentric coordinate
        T sum = T(1);
        for (size_t i = 0; i < n; ++i) {
            sum -= bary[i];
        }
        bary[n] = sum;

        return true;
    }

    // Special case for 1D: sort points and create segments between consecutive points
    void compute_delaunay_1d() {
        size_t num_points = npoints();

        // Create vector of (point_value, original_index) pairs
        std::vector<std::pair<T, size_t>> sorted_points(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            sorted_points[i] = {points_(i, 0), i};
        }

        // Sort by point value
        std::sort(sorted_points.begin(), sorted_points.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Create segments between consecutive sorted points
        simplices_.clear();
        simplices_.reserve((num_points - 1) * 2);
        for (size_t i = 0; i < num_points - 1; ++i) {
            simplices_.push_back(sorted_points[i].second);
            simplices_.push_back(sorted_points[i + 1].second);
        }
    }

    // Compute Delaunay triangulation using Bowyer-Watson algorithm
    void compute_delaunay() {
        size_t n = ndim_;
        size_t num_points = npoints();

        if (num_points < n + 1) {
            return; // Not enough points to form a simplex
        }

        // Special case for 1D: just sort points and create segments
        if (n == 1) {
            compute_delaunay_1d();
            return;
        }

        // Find bounding box and create super-simplex
        std::vector<T> min_coords(n, std::numeric_limits<T>::max());
        std::vector<T> max_coords(n, std::numeric_limits<T>::lowest());

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t d = 0; d < n; ++d) {
                min_coords[d] = std::min(min_coords[d], points_(i, d));
                max_coords[d] = std::max(max_coords[d], points_(i, d));
            }
        }

        // Compute center and make a large super-simplex
        std::vector<T> center(n);
        T max_range = T(0);
        for (size_t d = 0; d < n; ++d) {
            center[d] = (min_coords[d] + max_coords[d]) / T(2);
            max_range = std::max(max_range, max_coords[d] - min_coords[d]);
        }
        // Make super-simplex much larger than the data
        T M = std::max(max_range, T(1)) * T(100);

        // Create super-simplex vertices
        // For 2D: create a large triangle with vertices well outside the bounding box
        // For nD: create vertices along each axis direction plus one opposite
        std::vector<T> super_vertices((n + 1) * n);

        // Vertex 0: center + M in all positive directions
        for (size_t d = 0; d < n; ++d) {
            super_vertices[0 * n + d] = center[d] + M;
        }

        // Vertices 1 to n: each has one coordinate at center - M*n, others at center
        for (size_t i = 1; i <= n; ++i) {
            for (size_t d = 0; d < n; ++d) {
                if (d == i - 1) {
                    super_vertices[i * n + d] = center[d] - M * T(n);
                } else {
                    super_vertices[i * n + d] = center[d] + M;
                }
            }
        }

        // Store super-simplex as first simplex
        // Vertex indices for super-simplex: num_points, num_points+1, ..., num_points+n
        std::vector<std::vector<size_t>> triangulation;
        std::vector<size_t> super_simplex(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            super_simplex[i] = num_points + i;
        }
        triangulation.push_back(super_simplex);

        // Extended points array including super-vertices
        auto get_point = [&](size_t idx, size_t dim) -> T {
            if (idx < num_points) {
                return points_(idx, dim);
            } else {
                return super_vertices[(idx - num_points) * n + dim];
            }
        };

        // Circumsphere test - check if point is inside circumsphere of simplex
        // Uses explicit circumcenter computation for robustness
        auto in_circumsphere = [&](const std::vector<size_t>& simplex, size_t point_idx) -> bool {
            // Compute circumcenter by solving: 2*(v_i - v_0) · c = |v_i|² - |v_0|²
            // This gives us an n×n linear system

            // Get first vertex as reference
            std::vector<T> v0(n);
            T v0_sq = T(0);
            for (size_t d = 0; d < n; ++d) {
                v0[d] = get_point(simplex[0], d);
                v0_sq += v0[d] * v0[d];
            }

            // Build the linear system
            std::vector<T> mat(n * n);
            std::vector<T> rhs(n);

            for (size_t i = 0; i < n; ++i) {
                T vi_sq = T(0);
                for (size_t d = 0; d < n; ++d) {
                    T vi_d = get_point(simplex[i + 1], d);
                    mat[i * n + d] = T(2) * (vi_d - v0[d]);
                    vi_sq += vi_d * vi_d;
                }
                rhs[i] = vi_sq - v0_sq;
            }

            // Solve using Gaussian elimination with partial pivoting
            std::vector<size_t> perm(n);
            std::iota(perm.begin(), perm.end(), 0);

            for (size_t k = 0; k < n; ++k) {
                // Find pivot
                T max_val = std::abs(mat[perm[k] * n + k]);
                size_t max_idx = k;
                for (size_t i = k + 1; i < n; ++i) {
                    T val = std::abs(mat[perm[i] * n + k]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }

                if (max_val < eps_) {
                    return false; // Degenerate simplex
                }

                std::swap(perm[k], perm[max_idx]);

                // Eliminate
                for (size_t i = k + 1; i < n; ++i) {
                    T factor = mat[perm[i] * n + k] / mat[perm[k] * n + k];
                    for (size_t j = k + 1; j < n; ++j) {
                        mat[perm[i] * n + j] -= factor * mat[perm[k] * n + j];
                    }
                    rhs[perm[i]] -= factor * rhs[perm[k]];
                }
            }

            // Back substitution to get circumcenter
            std::vector<T> circumcenter(n);
            for (size_t k = n; k-- > 0;) {
                T sum = rhs[perm[k]];
                for (size_t j = k + 1; j < n; ++j) {
                    sum -= mat[perm[k] * n + j] * circumcenter[j];
                }
                circumcenter[k] = sum / mat[perm[k] * n + k];
            }

            // Compute circumradius squared (distance from circumcenter to v0)
            T radius_sq = T(0);
            for (size_t d = 0; d < n; ++d) {
                T diff = circumcenter[d] - v0[d];
                radius_sq += diff * diff;
            }

            // Check if test point is inside circumsphere
            T dist_sq = T(0);
            for (size_t d = 0; d < n; ++d) {
                T diff = get_point(point_idx, d) - circumcenter[d];
                dist_sq += diff * diff;
            }

            return dist_sq < radius_sq - eps_;
        };

        // Insert points one by one (Bowyer-Watson algorithm)
        for (size_t p = 0; p < num_points; ++p) {
            // Find all simplices whose circumsphere contains the new point
            std::vector<size_t> bad_simplices;
            for (size_t s = 0; s < triangulation.size(); ++s) {
                if (in_circumsphere(triangulation[s], p)) {
                    bad_simplices.push_back(s);
                }
            }

            // Find the boundary facets of the bad simplices cavity
            // A facet is on the boundary if it's only adjacent to one bad simplex
            std::vector<std::vector<size_t>> boundary_facets;

            for (size_t bad_idx : bad_simplices) {
                const auto& simplex = triangulation[bad_idx];
                // Each facet is formed by omitting one vertex
                for (size_t omit = 0; omit <= n; ++omit) {
                    std::vector<size_t> facet;
                    for (size_t v = 0; v <= n; ++v) {
                        if (v != omit) {
                            facet.push_back(simplex[v]);
                        }
                    }
                    std::sort(facet.begin(), facet.end());

                    // Check if this facet is shared with another bad simplex
                    bool shared = false;
                    for (size_t other_bad_idx : bad_simplices) {
                        if (other_bad_idx == bad_idx) continue;
                        const auto& other = triangulation[other_bad_idx];

                        std::vector<size_t> other_facet_sorted;
                        for (size_t v = 0; v <= n; ++v) {
                            other_facet_sorted.push_back(other[v]);
                        }
                        std::sort(other_facet_sorted.begin(), other_facet_sorted.end());

                        // Check all facets of other simplex
                        for (size_t other_omit = 0; other_omit <= n; ++other_omit) {
                            std::vector<size_t> other_facet;
                            for (size_t v = 0; v <= n; ++v) {
                                if (v != other_omit) {
                                    other_facet.push_back(other[v]);
                                }
                            }
                            std::sort(other_facet.begin(), other_facet.end());

                            if (facet == other_facet) {
                                shared = true;
                                break;
                            }
                        }
                        if (shared) break;
                    }

                    if (!shared) {
                        boundary_facets.push_back(facet);
                    }
                }
            }

            // Remove bad simplices (in reverse order to maintain indices)
            std::sort(bad_simplices.rbegin(), bad_simplices.rend());
            for (size_t bad_idx : bad_simplices) {
                triangulation.erase(triangulation.begin() + bad_idx);
            }

            // Create new simplices by connecting boundary facets to new point
            for (const auto& facet : boundary_facets) {
                std::vector<size_t> new_simplex = facet;
                new_simplex.push_back(p);
                triangulation.push_back(new_simplex);
            }
        }

        // Remove simplices that contain super-simplex vertices
        std::vector<std::vector<size_t>> final_triangulation;
        for (const auto& simplex : triangulation) {
            bool has_super_vertex = false;
            for (size_t v : simplex) {
                if (v >= num_points) {
                    has_super_vertex = true;
                    break;
                }
            }
            if (!has_super_vertex) {
                final_triangulation.push_back(simplex);
            }
        }

        // Flatten to simplices_ array
        simplices_.clear();
        simplices_.reserve(final_triangulation.size() * (n + 1));
        for (const auto& simplex : final_triangulation) {
            for (size_t v : simplex) {
                simplices_.push_back(v);
            }
        }
    }
};


#endif // LINEAR_ND_INTERPOLATOR_HPP