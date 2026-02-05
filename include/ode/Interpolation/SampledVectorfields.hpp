#ifndef SAMPLED_VECTORFIELDS_HPP
#define SAMPLED_VECTORFIELDS_HPP

#include "GridInterp.hpp"

namespace ode {


template<typename T, size_t NDIM>
class SampledVectorField : public RegularGridInterpolator<T, NDIM>{

    using Base = RegularGridInterpolator<T, NDIM>;

public:

    template<typename... Args>
    SampledVectorField(const Args&... args);

    auto                                    ode_func_norm() const;
    auto                                    ode_func() const;
    OdeResult<T>                            streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method) const;

    inline std::vector<Array2D<T, NDIM, 0>> streamplot_data(const T& max_length, const T& ds, size_t density) const;

private:

    template<size_t... I>
    std::vector<Array2D<T, NDIM, 0>> streamplot_data_core(const T& max_length, const T& ds, size_t density, std::index_sequence<I...>) const;

};




// ============================================================================
// IMPLEMENTATIONS
// ============================================================================




// ----------------------------------------------------------------------------
// SampledVectorField
// ----------------------------------------------------------------------------

template<typename T, size_t NDIM>
template<typename... Args>
SampledVectorField<T, NDIM>::SampledVectorField(const Args&... args) : Base(args...) {
    static_assert(sizeof...(args) == 2*NDIM, "SampledVectorField constructor requires NDIM grid arrays and NDIM field arrays");
}

template<typename T, size_t NDIM>
auto SampledVectorField<T, NDIM>::ode_func_norm() const {
    return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
        for (size_t i = 0; i < NDIM; i++) {
            if (!this->value_in_axis(q[i], i)) {
                for (size_t j = 0; j < NDIM; j++) {
                    out[j] = 0;
                }
                return;
            }
        }
        this->get_norm(out, q);
    };
}

template<typename T, size_t NDIM>
auto SampledVectorField<T, NDIM>::ode_func() const {
    return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
        for (size_t i = 0; i < NDIM; i++) {
            if (!this->value_in_axis(q[i], i)) {
                for (size_t j = 0; j < NDIM; j++){
                    out[j] = 0;
                }
                return;
            }
        }
        this->get(out, q);
    };
}

template<typename T, size_t NDIM>
OdeResult<T> SampledVectorField<T, NDIM>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method) const{
    /*
    direction: +1 forward, -1 backward
    */

    ODE<T> ode(OdeData{.rhs=this->ode_func(), .obj=this}, 0, x0, NDIM, rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);

    return ode.integrate(length, t_eval);
}

template<typename T, size_t NDIM>
inline std::vector<Array2D<T, NDIM, 0>> SampledVectorField<T, NDIM>::streamplot_data(const T& max_length, const T& ds, size_t density) const{
    return streamplot_data_core(max_length, ds, density, std::make_index_sequence<NDIM>{});
}

template<typename T, size_t NDIM>
template<size_t... I>
std::vector<Array2D<T, NDIM, 0>> SampledVectorField<T, NDIM>::streamplot_data_core(const T& max_length, const T& ds, size_t density, std::index_sequence<I...>) const{

    assert(max_length > ds && max_length > 0 && ds > 0 && "max_length and ds must be positive, and max_length must be greater than ds");
    assert(density > 1 && "Density must be greater than 1");
    auto ode = this->ode_func_norm();

    using Solver = RK4<T, NDIM, SolverPolicy::Static, decltype(ode), std::nullptr_t>;

    const std::array<Array1D<T>, NDIM>& X = this->x_all();
    std::array<size_t, NDIM> N;
    std::array<T, NDIM> L;
    std::array<T, NDIM> Dx;
    for (size_t axis=0; axis<NDIM; axis++){
        N[axis] = density;
        L[axis] = this->length(axis);
        Dx[axis] = L[axis] / (N[axis] - 1);
    }

    size_t max_pts = size_t(max_length / ds) + 1; //per integration direction

    NdArray<bool, NDIM> is_full(N.data(), NDIM);
    std::array<std::vector<int>, NDIM> reached; // reached[axis] = vector of reached indices along that axis
    is_full.set(false);

    T min_length = 0.2 * (*std::min_element(L.begin(), L.end())); // minimum length of streamline to be kept, as a fraction of the minimum grid length

    std::array<int, NDIM> i_start;
    std::array<int, NDIM> i_curr;

    auto GetIdx = [&](std::array<int, NDIM>& i, const T* x) LAMBDA_INLINE{
        ((i[I] = size_t(std::round((x[I] - X[I][0]) / Dx[I]))), ...);
    };

    auto InBounds = [&](const std::array<int, NDIM>& i) LAMBDA_INLINE{
        return ((i[I] >= 0 && i[I] < int(N[I])) && ...);
    };

    auto get_ds = [](const T* q_old, const T* q_new) LAMBDA_INLINE{
        return sqrt((((q_new[I] - q_old[I])* (q_new[I] - q_old[I])) + ...));
    };

    auto IntegrateDirection = [&](Solver& solver, T& s_total, T* const * x, size_t& n_steps_tot, const int dir){
        // x[I] are preallocated arrays of size max_pts + 1. x[I][0] = x0[I], so at each step, we write to x[I][step], starting from x[I][1]
        std::array<T, NDIM> ics = {x[I][0]...};
        assert(this->coords_in_bounds(ics[I]...) && "Initial point out of bounds");
        assert(dir == 1 || dir == -1 && "Direction must be either +1 or -1");

        size_t n_steps = 0;
        solver.set_ics(0, ics.data(), ds, dir);
        const T* q_new;
        const T* q_old;


        GetIdx(i_start, ics.data());
        while ((n_steps < max_pts) && solver.advance()){
            q_new = solver.vector().data();
            q_old = solver.vector_old().data();
            GetIdx(i_curr, q_new);
            s_total += get_ds(q_old, q_new);
            n_steps++;
            n_steps_tot++;
            ((x[I][n_steps*dir] = q_new[I]), ...);
            if (((i_curr[I] != i_start[I]) || ...) && InBounds(i_curr)){
                if(!is_full(i_curr[I]...)){
                    is_full(i_curr[I]...) = true;
                    (reached[I].push_back(i_curr[I]), ...);
                    ((i_start[I] = i_curr[I]), ...);
                }else {
                    break;
                }
            }
        }
        return n_steps;
    };


    auto GetStreamline = [&](bool& success, Array2D<T, NDIM, 0>& worker_line, T* x0) {
        assert(worker_line.shape(1) == 2*max_pts + 1 && "Line array must have shape (2, 2*max_pts + 1)");
        ((worker_line(I, max_pts) = x0[I]), ...);

        size_t n_steps_tot = 0;
        Solver solver({.rhs=ode}, 0, nullptr, NDIM, 1e-5, 1e-5, 0, inf<T>(), ds, +1);

        T s_total = 0;
        ((reached[I].clear()), ...);

        std::array<T*, NDIM> x = {worker_line.ptr(I, max_pts)...};
        IntegrateDirection(solver, s_total, x.data(), n_steps_tot, +1);
        size_t n_steps_bwd = IntegrateDirection(solver, s_total, x.data(), n_steps_tot, -1);

        // if the streamline is long enough and n_steps_tot  > 1, we keep it:
        std::array<const T*, NDIM> x_line = {worker_line.ptr(I, max_pts - n_steps_bwd)...};
        Array2D<T, NDIM, 0> true_line(NDIM, n_steps_tot);
        if (s_total > min_length && n_steps_tot > 1){
            ((copy_array(true_line.ptr(I, 0), x_line[I], n_steps_tot)), ...);

            //mark the initial point
            GetIdx(i_start, x0);
            if (InBounds(i_start)){
                is_full(i_start[I]...) = true;
            }
            success = true;
        }else{
            // unmark the points we marked during this integration attempt
            for (size_t k = 0; k < reached[0].size(); k++){
                is_full(reached[I][k]...) = false;
            }
            success = false;
        }
        return true_line;
    };

    std::vector<Array2D<T, NDIM, 0>> streamlines;
    auto TryTrajectory = [&](Array2D<T, NDIM, 0>& worker_line, const std::array<int, NDIM>& idx) LAMBDA_INLINE{

        if (((idx[I] < 0 || idx[I] >= int(N[I])) || ...)) {
            return;
        }
        std::array<T, NDIM> x0 = {(X[I][0] + idx[I] * Dx[I])...};
        assert(this->coords_in_bounds(x0[I]...) && "Initial point out of bounds");
        if (!is_full(idx[I]...)){
            bool success;
            auto new_line = GetStreamline(success, worker_line, x0.data());
            if (success) {
                streamlines.push_back(std::move(new_line));
            }
        }
    };

    // Seeding streamlines from edges working inwards (shell by shell):
    Array2D<T, NDIM, 0> line(NDIM, 2*max_pts + 1);

    size_t min_N = *std::min_element(N.begin(), N.end());
    size_t max_shells = (min_N + 1) / 2;

    // Compute shell level: minimum distance to any boundary
    auto shell_level = [&N](const std::array<int, NDIM>& idx) -> size_t {
        size_t min_dist = SIZE_MAX;
        for (size_t d = 0; d < NDIM; d++) {
            size_t dist = std::min(size_t(idx[d]), N[d] - 1 - idx[d]);
            min_dist = std::min(min_dist, dist);
        }
        return min_dist;
    };

    for (size_t shell = 0; shell < max_shells; shell++) {
        // Iterate over all grid points within the current shell's bounding box
        std::array<int, NDIM> idx;
        std::fill(idx.begin(), idx.end(), int(shell));
        while (true) {
            // Only process points that are exactly on this shell boundary
            if (shell_level(idx) == shell) {
                TryTrajectory(line, idx);
            }

            // Increment multi-dimensional index
            size_t d = 0;
            while (d < NDIM) {
                idx[d]++;
                if (idx[d] < int(N[d]) - int(shell)) {
                    break;
                }
                idx[d] = int(shell);
                d++;
            }
            if (d == NDIM) {break;} // All combinations exhausted
        }
    }
    return streamlines;
}

} // namespace ode


#endif // SAMPLED_VECTORFIELDS_HPP