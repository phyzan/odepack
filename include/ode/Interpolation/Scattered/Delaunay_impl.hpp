#ifndef DELAUNAY_IMPL_HPP
#define DELAUNAY_IMPL_HPP


#include "Delaunay.hpp"
#include "../../Tools.hpp"
#include <libqhull_r/qhull_ra.h>

namespace ode{

template<size_t NDIM>
DelaunayTri<NDIM>::DelaunayTri(const double* points, size_t n_points, size_t ndim) : points_(points, n_points, ndim) {
    compute_delaunay();

    int ND = this->ndim();
    v0_.resize(nsimplices(), ND);
    invT_.resize(nsimplices(), ND, ND);

    constexpr Allocation Alloc = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);
    Array2D<double, NDIM, NDIM, Alloc> A(ND, ND);

    Array1D<double, NDIM, Alloc> work(ND);
    Array1D<size_t, NDIM, Alloc> pivot(ND);

    for (int s = 0; s < nsimplices(); ++s) {
        const int* v = get_simplex(s);

        // v0
        for (int i = 0; i < ND; ++i){
            v0_(s, i) = points_(v[ND], i);
        }
        // build edge matrix

        for (int i = 0; i < ND; ++i){
            for (int j = 0; j < ND; ++j){
                A(i, j) = points_(v[j], i) - v0_(s, i);
            }
        }

        // invert A ONCE (reuse your existing solver here)
        inv_mat_row_major(invT_.ptr(s, 0, 0), A.data(), ND, work.data(), pivot.data());

    }

}


template<size_t NDIM>
int DelaunayTri<NDIM>::ndim() const { return points_.Ncols(); }

template<size_t NDIM>
int DelaunayTri<NDIM>::npoints() const { return points_.Nrows(); }

template<size_t NDIM>
const Array2D<double, 0, NDIM>& DelaunayTri<NDIM>::get_points() const { return points_; }

template<size_t NDIM>
const Array2D<int, 0, DelaunayTri<NDIM>::DIM_SPX>& DelaunayTri<NDIM>::get_simplices() const { return simplices_; }

template<size_t NDIM> const Array2D<int, 0, DelaunayTri<NDIM>::DIM_SPX>& DelaunayTri<NDIM>::get_neighbors() const { return neighbors_; }

template<size_t NDIM>
const Array2D<double, 0, NDIM>& DelaunayTri<NDIM>::get_v0() const { return v0_; }

template<size_t NDIM> const Array3D<double, 0, NDIM, NDIM>& DelaunayTri<NDIM>::get_invT() const {
    return invT_; 
}

template<size_t NDIM>
int DelaunayTri<NDIM>::nsimplices() const { return simplices_.Nrows(); }

template<size_t NDIM>
const int* DelaunayTri<NDIM>::get_simplex(int simplex_idx) const {
    assert(simplex_idx >= 0 && simplex_idx < nsimplices() && "Simplex index out of bounds");
    return simplices_.ptr(simplex_idx, 0);
}

template<size_t NDIM>
const int* DelaunayTri<NDIM>::get_neighbors(int simplex_idx) const {
    assert(simplex_idx >= 0 && simplex_idx < nsimplices() && "Simplex index out of bounds");
    return neighbors_.ptr(simplex_idx, 0);
}

template<size_t NDIM>
int DelaunayTri<NDIM>::find_simplex(const double* point) const {
    int num_simplices = nsimplices();    
    int nd = ndim();
    if (num_simplices == 0 || !all_are_finite(point, nd)) {
        return -1;
    }

    if (nd == 1) {
        const double x = point[0];
        for (int sidx = 0; sidx < num_simplices; ++sidx) {
            const int* v = get_simplex(sidx);
            const double x0 = points_(v[0], 0);
            const double x1 = points_(v[1], 0);
            const double xmin = std::min(x0, x1);
            const double xmax = std::max(x0, x1);
            if (x >= xmin - EPS && x <= xmax + EPS) {
                return sidx;
            }
        }
        return -1;
    }


    Array1D<double, DIM_SPX, SPX_ALLOC> bary(nd + 1);

    // Walking algorithm: start from cached simplex and walk towards the point
    int& last_simplex = this->thread_cache(); // start from the last simplex found by this thread on this object
    int s = (last_simplex >= 0 && last_simplex < num_simplices) ? last_simplex : 0;
    int prev = -1; // track previous simplex to avoid oscillation
    int max_iter = num_simplices; // prevent infinite loops

    auto is_inside = [&](int simplex_idx) {
        if (!compute_barycentric(bary.data(), simplex_idx, point)) {
            return false;
        }
        for (int i = 0; i <= nd; ++i) {
            if (bary[i] < -EPS) {
                return false;
            }
        }
        return true;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        if (!compute_barycentric(bary.data(), s, point)) {
            break;
        }

        // Find most negative barycentric coordinate (with and without prev-skip)
        int min_idx = -1;
        double min_val = -EPS;
        int min_idx_any = -1;
        double min_val_any = -EPS;
        bool any_negative = false;

        for (int i = 0; i <= nd; ++i) {
            if (bary[i] < -EPS) {
                any_negative = true;
            }

            if (bary[i] < min_val_any) {
                min_val_any = bary[i];
                min_idx_any = i;
            }

            // Skip if this neighbor is the previous simplex (avoid oscillation)
            if (neighbors_(s, i) == prev) {
                continue;
            }
            if (bary[i] < min_val) {
                min_val = bary[i];
                min_idx = i;
            }
        }

        if (!any_negative) {
            // All coordinates >= -EPS, point is inside this simplex
            last_simplex = s; // cache for next query
            return s;
        }

        if (min_idx == -1) {
            // Only the previous neighbor is a candidate; fall back to the most negative face.
            min_idx = min_idx_any;
        }

        int next = (min_idx >= 0) ? neighbors_(s, min_idx) : -1;
        if (next == -1) {
            // Hit boundary face with negative barycentric coord -> point is outside convex hull
            return -1;
        }else if (next == s) {
            break;  // Degenerate case, fall back to brute force
        }
        prev = s;
        s = next;
    }

    // Fallback: brute force check in case the walk stalls (degenerate/simplex mapping issues).
    for (int sidx = 0; sidx < num_simplices; ++sidx) {
        if (is_inside(sidx)) {
            last_simplex = sidx;
            return sidx;
        }
    }

    return -1;
}

template<size_t NDIM>
bool DelaunayTri<NDIM>::compute_barycentric(double* out, int simplex_idx, const double* point) const{
    const double* v0 = v0_.ptr(simplex_idx, 0);
    const double* T  = invT_.ptr(simplex_idx, 0, 0);

    int ND = ndim();

    constexpr Allocation Alloc = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);
    Array1D<double, NDIM, Alloc> y(ND);
    for (int i = 0; i < ND; ++i){
        y[i] = point[i] - v0[i];
    }

    // bary[1..NDIM]
    for (int i = 0; i < ND; ++i) {
        double sum = 0.0;
        for (int j = 0; j < ND; ++j){
            sum += T[i * ND + j] * y[j];
        }
        out[i] = sum;
    }

    // bary[0]
    double sumb = 1.0;
    for (int i = 0; i < ND; ++i){
        sumb -= out[i];
    }
    out[ND] = sumb;

    return all_are_finite(out, ND + 1);

};

template<size_t NDIM>
bool DelaunayTri<NDIM>::interpolate(double* out, const double* point, const double* field, int n_fields) const {
    int simplex_idx = find_simplex(point);
    if (simplex_idx == -1) {
        std::fill(out, out + n_fields, std::numeric_limits<double>::quiet_NaN());
        return false;
    }
    

    int nd = int(ndim());
    Array1D<double, DIM_SPX, SPX_ALLOC> bary(nd + 1);
    compute_barycentric(bary.data(), simplex_idx, point);
    const int* vertices = get_simplex(simplex_idx);

    View<double, Layout::C, 0, 0> field_view(field, npoints(), n_fields);
    std::fill(out, out + n_fields, 0.0);
    for (int i=0; i <= ndim(); ++i){
        for (int f = 0; f < n_fields; ++f) {
            out[f] += bary[i] * field_view(vertices[i], f);
        }
    }

    return true;
}

template<size_t NDIM>
double DelaunayTri<NDIM>::get_value(const double* point, const double* field) const {
    double res;
    interpolate(&res, point, field, 1);
    return res;
}

template<size_t NDIM>
void DelaunayTri<NDIM>::compute_delaunay_1d() {
    int num_points = npoints();
    int num_simplices = num_points - 1;

    std::vector<std::pair<double, int>> sorted_points(num_points);
    for (int i = 0; i < num_points; ++i) {
        sorted_points[i] = {points_(i, 0), i};
    }

    std::sort(sorted_points.begin(), sorted_points.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    // Build simplices
    simplices_.resize(num_simplices, 2);
    for (int i = 0; i < num_simplices; ++i) {
        simplices_(i, 0) = sorted_points[i].second;
        simplices_(i, 1) = sorted_points[i + 1].second;
    }

    // Build neighbors: neighbor[0] is opposite vertex[0], neighbor[1] is opposite vertex[1]
    // In 1D: opposite the left vertex is the simplex to the right, and vice versa.
    neighbors_.resize(num_simplices, 2);
    for (int i = 0; i < num_simplices; ++i) {
        neighbors_(i, 0) = (i < num_simplices - 1) ? (i + 1) : -1;    // right neighbor (opposite left vertex)
        neighbors_(i, 1) = (i > 0) ? (i - 1) : -1;                    // left neighbor (opposite right vertex)
    }
}


template<size_t NDIM>
void DelaunayTri<NDIM>::compute_delaunay_nd() {
    int n = int(ndim());
    int num_points = npoints();

    const Array2D<double, 0, NDIM>& coords = get_points();

    // Initialize Qhull
    qhT qh_qh;
    qhT* qh = &qh_qh;
    QHULL_LIB_CHECK

    qh_zero(qh, nullptr);

    // Run Delaunay triangulation
    // "d" = Delaunay, "Qt" = triangulated output, "Qbb" = scale to unit box, "Qz" = add point at infinity
    char flags[] = "qhull d Qt Qbb Qz";
    int exitcode = qh_new_qhull(qh, n, num_points,
                                const_cast<double*>(coords.data()), false, flags, nullptr, nullptr);

    if (exitcode != 0) {
        qh_freeqhull(qh, !qh_ALL);
        int curlong, totlong;
        qh_memfreeshort(qh, &curlong, &totlong);
        throw std::runtime_error("Qhull Delaunay triangulation failed");
    }

    // First pass: build facet pointer -> index map and collect simplices
    std::unordered_map<facetT*, int> facet_to_idx;
    std::vector<facetT*> facet_list;
    std::vector<int> simplex_array;

    facetT* facet;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            vertexT* vertex;
            vertexT** vertexp;

            std::vector<int> simplex_verts;
            simplex_verts.reserve(n + 1);

            FOREACHvertex_(facet->vertices) {
                int point_idx = qh_pointid(qh, vertex->point);
                if (point_idx >= 0 && point_idx < num_points) {
                    simplex_verts.push_back(point_idx);
                }
            }

            if (simplex_verts.size() == size_t(n + 1)) {
                int idx = int(facet_list.size());
                facet_to_idx[facet] = idx;
                facet_list.push_back(facet);
                for (int v : simplex_verts) {
                    simplex_array.push_back(v);
                }
            }
        }
    }

    int num_simplices = int(facet_list.size());
    simplices_ = Array2D<int, 0, DIM_SPX>(simplex_array.data(), num_simplices, n + 1);

    // Second pass: extract neighbors
    neighbors_.resize(num_simplices, n + 1);
    neighbors_.set(-1); // default to -1 (no neighbor)

    for (int s = 0; s < num_simplices; ++s) {
        facetT* f = facet_list[s];
        facetT** neighborp;
        facetT* neighbor;
        int i = 0;

        FOREACHneighbor_(f) {
            if (neighbor && !neighbor->upperdelaunay) {
                auto it = facet_to_idx.find(neighbor);
                if (it != facet_to_idx.end()) {
                    neighbors_(s, i) = it->second;
                }
            }
            ++i;
        }
    }

    // Cleanup Qhull
    qh_freeqhull(qh, !qh_ALL);
    int curlong, totlong;
    qh_memfreeshort(qh, &curlong, &totlong);
}


template<size_t NDIM>
void DelaunayTri<NDIM>::compute_delaunay() {
    int n = ndim();
    int num_points = npoints();

    if (num_points < n + 1) {return;}

    if (n == 1) {
        compute_delaunay_1d();
    } else {
        compute_delaunay_nd();
    }
}

template<size_t NDIM>
int& DelaunayTri<NDIM>::thread_cache() const {
    // each thread keeps track of the last simplex it found a point in, to speed up subsequent queries
    // each spawned object will have its own cache.
    // So it is a per-thread cache, but also per-object cache.
    // This is so that the interpolation of each sampled function from multiple threads is safe and efficient.
    thread_local std::unordered_map<const DelaunayTri*, int> cache;
    return cache[this];
}

template<size_t NDIM>
void DelaunayTri<NDIM>::set_state(const View2D<double, 0, NDIM>& points, const View2D<int, 0, DIM_SPX>& simplices, const View2D<int, 0, DIM_SPX>& neighbors, const View2D<double, 0, NDIM>& v0, const View3D<double, 0, NDIM, NDIM>& invT){
    points_.resize(points.shape(), points.ndim());
    copy_array(points_.data(), points.data(), points.size());

    simplices_.resize(simplices.shape(), simplices.ndim());
    copy_array(simplices_.data(), simplices.data(), simplices.size());

    neighbors_.resize(neighbors.shape(), neighbors.ndim());
    copy_array(neighbors_.data(), neighbors.data(), neighbors.size());

    v0_.resize(v0.shape(), v0.ndim());
    copy_array(v0_.data(), v0.data(), v0.size());
    
    invT_.resize(invT.shape(), invT.ndim());
    copy_array(invT_.data(), invT.data(), invT.size());
}


} // namespace ode


#endif // DELAUNAY_IMPL_HPP