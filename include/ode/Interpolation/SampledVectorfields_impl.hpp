#ifndef SAMPLED_VECTORFIELDS_IMPL_HPP
#define SAMPLED_VECTORFIELDS_IMPL_HPP


#include "SampledVectorfields.hpp"

namespace ode{


// ----------------------------------------------------------------------------
// SampledVectorField
// ----------------------------------------------------------------------------

template<typename T, size_t NDIM>
SampledVectorField<T, NDIM>::SampledVectorField(const T* values, std::vector<View1D<T>> grid, bool values_axis_first) : Base(values, grid.size(), grid, values_axis_first) {}

template<typename T, size_t NDIM>
SampledVectorField<T, NDIM>::SampledVectorField(const T* values, std::vector<MutView1D<T>> grid, bool values_axis_first) : Base(values, grid.size(), grid, values_axis_first) {}

template<typename T, size_t NDIM>
OdeResult<T> SampledVectorField<T, NDIM>::streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method) const{
    /*
    direction: +1 forward, -1 backward
    */

    ODE<T> ode(OdeData<Func<T>, void>{.rhs=ode_func, .obj=this}, 0, x0, NDIM, rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);

    return ode.integrate(length, t_eval);
}

template<typename T, size_t NDIM>
 std::vector<Array2D<T, NDIM, 0>> SampledVectorField<T, NDIM>::streamplot_data(const T& max_length, const T& ds, size_t density) const{
    return streamplot_data_core(max_length, ds, density, std::make_index_sequence<NDIM>{});
}

template<typename T, size_t NDIM>
template<size_t... I>
std::vector<Array2D<T, NDIM, 0>> SampledVectorField<T, NDIM>::streamplot_data_core(const T& max_length, const T& ds, size_t density, std::index_sequence<I...>) const{

    assert(max_length > ds && max_length > 0 && ds > 0 && "max_length and ds must be positive, and max_length must be greater than ds");
    assert(density > 1 && "Density must be greater than 1");

    // using RK4<T, 0, RichVirtual> because it is the Python bindings instanciate these templates parameters,
    // event though they are not the optimal ones. Ideally we want RK4<T, NDIM, Static, decltype(lambda_func), std::nullptr_t>, but the performance
    // gain is far too small to justify the increase in binary size and compilation time.
    using Solver = RK4<T, 0, SolverPolicy::RichVirtual, Func<T>, void>;

    std::vector<T> args{};
    std::vector<const Event<T>*> events{};
    Solver solver(OdeData<Func<T>, void>{.rhs=ode_func_norm, .obj=this}, 0, nullptr, this->ndim(), 1e-5, 1e-5, 0, inf<T>(), ds, +1, static_cast<const std::vector<T>&>(args), static_cast<const std::vector<const Event<T>*>&>(events));

    const auto& X = this->x_all();
    Array1D<size_t, NDIM, Alloc> N(this->ndim());
    Array1D<T, NDIM, Alloc> L(this->ndim());
    Array1D<T, NDIM, Alloc> Dx(this->ndim());
    for (size_t axis=0; axis<this->ndim(); axis++){
        N[axis] = density;
        L[axis] = this->length(axis);
        Dx[axis] = L[axis] / (N[axis] - 1);
    }


    size_t max_pts = size_t(max_length / ds) + 1; //per integration direction

    NdArray<bool, NDIM> is_full(N.data(), this->ndim());
    Array1D<std::vector<int>, NDIM, Alloc> reached(this->ndim()); // reached[axis] = vector of reached indices along that axis
    is_full.set(false);

    T min_length = 0.2 * (*std::min_element(L.begin(), L.end())); // minimum length of streamline to be kept, as a fraction of the minimum grid length

    Array1D<int, NDIM, Alloc> i_start(this->ndim());
    Array1D<int, NDIM, Alloc> i_curr(this->ndim());

    auto GetIdx = [&](Array1D<int, NDIM, Alloc>& i, const T* x) LAMBDA_INLINE{
        if constexpr (NDIM > 0){
            ((i[I] = size_t(std::round((x[I] - X[I][0]) / Dx[I]))), ...);
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                i[axis] = size_t(std::round((x[axis] - X[axis][0]) / Dx[axis]));
            }
        }
    };

    Array1D<int, NDIM, Alloc> idx_aux(this->ndim());

    auto InBounds = [&](const Array1D<int, NDIM, Alloc>& i) LAMBDA_INLINE{
        if constexpr (NDIM > 0){
            return ((i[I] >= 0 && i[I] < int(N[I])) && ...);
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                if (i[axis] < 0 || i[axis] >= int(N[axis])){
                    return false;
                }
            }
            return true;
        }
    };

    auto get_ds = [this](const T* q_old, const T* q_new) LAMBDA_INLINE{
        if constexpr (NDIM > 0){
            return sqrt((((q_new[I] - q_old[I])* (q_new[I] - q_old[I])) + ...));
        }else{
            T sum = 0;
            for (size_t axis=0; axis<this->ndim(); axis++){
                sum += (q_new[axis] - q_old[axis]) * (q_new[axis] - q_old[axis]);
            }
            return sqrt(sum);
        }
    };

    auto IntegrateDirection = [&](Solver& solver, T& s_total, T* const * x, size_t& n_steps_tot, const int dir) -> size_t {
        // x[I] are preallocated arrays of size max_pts + 1. x[I][0] = x0[I], so at each step, we write to x[I][step], starting from x[I][1]
        Array1D<T, NDIM, Alloc> ics(this->ndim());
        if constexpr (NDIM > 0){
            ics = Array1D<T, NDIM, Alloc>{x[I][0]...};
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                ics[axis] = x[axis][0];
            }
        }
        
        assert(this->coords_in_bounds(ics.data()) && "Initial point out of bounds");
        assert((dir == 1 || dir == -1) && "Direction must be either +1 or -1");
        if (!solver.set_ics(0, ics.data(), ds, dir)) {
            return 0;
        }
        size_t n_steps = 0;
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
            if constexpr (NDIM > 0){
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
            }else{
                for (size_t axis=0; axis<this->ndim(); axis++){
                    x[axis][n_steps*dir] = q_new[axis];
                }
                if ((!allEqual(i_start.data(), i_curr.data(), this->ndim())) && InBounds(i_curr)){
                    if(!is_full.getElem(i_curr.data())){
                        is_full.getElem(i_curr.data()) = true;
                        for (size_t axis=0; axis<this->ndim(); axis++){
                            reached[axis].push_back(i_curr[axis]);
                            i_start[axis] = i_curr[axis];
                        }
                    }else {
                        break;
                    }
                }
            }

        }
        return n_steps;
    };


    auto GetStreamline = [&](bool& success, Array2D<T, NDIM, 0>& worker_line, const T* x0) {
        assert(worker_line.shape(1) == 2*max_pts + 1 && "Line array must have shape (2, 2*max_pts + 1)");
        if constexpr (NDIM > 0) {
            ((worker_line(I, max_pts) = x0[I]), ...);
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                worker_line(axis, max_pts) = x0[axis];
            }
        }
        

        size_t n_steps_tot = 0;

        // The args and events are passed as const & because this exact signature is compiled in the
        // Python bindings, so this saves binary size and compilation time.

        T s_total = 0;
        Array1D<T*, NDIM, Alloc> x(this->ndim());
        if constexpr (NDIM > 0){
            x = {worker_line.ptr(I, max_pts)...};
            ((reached[I].clear()), ...);
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                x[axis] = worker_line.ptr(axis, max_pts);
                reached[axis].clear();
            }
        }

        IntegrateDirection(solver, s_total, x.data(), n_steps_tot, +1);
        size_t n_steps_bwd = IntegrateDirection(solver, s_total, x.data(), n_steps_tot, -1);

        // if the streamline is long enough and n_steps_tot  > 1, we keep it:
        Array1D<const T*, NDIM, Alloc> x_line(this->ndim());
        if constexpr (NDIM > 0){
            x_line = {worker_line.ptr(I, max_pts - n_steps_bwd)...};
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                x_line[axis] = worker_line.ptr(axis, max_pts - n_steps_bwd);
            }
        }
        Array2D<T, NDIM, 0> true_line(this->ndim(), n_steps_tot);
        if (s_total > min_length && n_steps_tot > 1){
            if constexpr (NDIM > 0){
                ((copy_array(true_line.ptr(I, 0), x_line[I], n_steps_tot)), ...);
            }else{
                for (size_t axis=0; axis<this->ndim(); axis++){
                    copy_array(true_line.ptr(axis, 0), x_line[axis], n_steps_tot);
                }
            }

            //mark the initial point
            GetIdx(i_start, x0);
            if (InBounds(i_start)){
                if constexpr (NDIM > 0){
                    is_full(i_start[I]...) = true;
                }else{
                    is_full.getElem(i_start.data()) = true;
                }
                
            }
            success = true;
        }else{
            // unmark the points we marked during this integration attempt
            for (size_t k = 0; k < reached[0].size(); k++){
                if constexpr (NDIM > 0){
                    is_full(reached[I][k]...) = false;
                }else{
                    for (size_t axis=0; axis<this->ndim(); axis++){
                        idx_aux[axis] = reached[axis][k];
                    }
                    is_full.getElem(idx_aux.data()) = false;
                }
            }
            success = false;
        }
        return true_line;
    };

    std::vector<Array2D<T, NDIM, 0>> streamlines;
    auto TryTrajectory = [&](Array2D<T, NDIM, 0>& worker_line, const Array1D<int, NDIM, Alloc>& idx) LAMBDA_INLINE{

        if constexpr (NDIM > 0) {
            if (((idx[I] < 0 || idx[I] >= int(N[I])) || ...)) {
                return;
            }
        } else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                if (idx[axis] < 0 || idx[axis] >= int(N[axis])){
                    return;
                }
            }
        }
        Array1D<T, NDIM, Alloc> x0(this->ndim());
        if constexpr (NDIM > 0){
            x0 = Array1D<T, NDIM, Alloc>{(X[I][0] + idx[I] * Dx[I])...};
        }else{
            for (size_t axis=0; axis<this->ndim(); axis++){
                x0[axis] = X[axis][0] + idx[axis] * Dx[axis];
            }
        }

        assert(this->coords_in_bounds(x0.data()) && "Initial point out of bounds");
        if (!is_full.getElem(idx.data())) {
            bool success;
            auto new_line = GetStreamline(success, worker_line, x0.data());
            if (success) {
                streamlines.push_back(std::move(new_line));
            }
        }
    };

    // Seeding streamlines from edges working inwards (shell by shell):
    Array2D<T, NDIM, 0> line(this->ndim(), 2*max_pts + 1);

    size_t min_N = *std::min_element(N.begin(), N.end());
    size_t max_shells = (min_N + 1) / 2;

    // Compute shell level: minimum distance to any boundary
    auto shell_level = [&N, this](const Array1D<int, NDIM, Alloc>& idx) -> size_t {
        size_t min_dist = SIZE_MAX;
        for (size_t d = 0; d < this->ndim(); d++) {
            size_t dist = std::min(size_t(idx[d]), N[d] - 1 - idx[d]);
            min_dist = std::min(min_dist, dist);
        }
        return min_dist;
    };

    for (size_t shell = 0; shell < max_shells; shell++) {
        // Iterate over all grid points within the current shell's bounding box
        Array1D<int, NDIM, Alloc> idx(this->ndim());
        std::fill(idx.begin(), idx.end(), int(shell));
        while (true) {
            // Only process points that are exactly on this shell boundary
            if (shell_level(idx) == shell) {
                TryTrajectory(line, idx);
            }

            // Increment multi-dimensional index
            size_t d = 0;
            while (d < this->ndim()) {
                idx[d]++;
                if (idx[d] < int(N[d]) - int(shell)) {
                    break;
                }
                idx[d] = int(shell);
                d++;
            }
            if (d == this->ndim()) {break;} // All combinations exhausted
        }
    }
    return streamlines;
}



template<typename T, size_t NDIM>
void SampledVectorField<T, NDIM>::ode_func_norm(T* out, const T& t, const T* q, const T* args, const void* ptr){
    assert(ptr != nullptr && "pointer is null");
    const auto* self = reinterpret_cast<const SampledVectorField<T, NDIM>*>(ptr);
    size_t nd = self->ndim();
    for (size_t i = 0; i < nd; i++) {
        if (!self->value_in_axis(q[i], i)) {
            for (size_t j = 0; j < nd; j++) {
                out[j] = 0;
            }
            return;
        }
    }
    self->interp_norm(out, q);
};

template<typename T, size_t NDIM>
void SampledVectorField<T, NDIM>::ode_func(T* out, const T& t, const T* q, const T* args, const void* ptr){
    assert(ptr != nullptr && "pointer is null");
    const auto* self = reinterpret_cast<const SampledVectorField<T, NDIM>*>(ptr);
    size_t nd = self->ndim();
    for (size_t i = 0; i < nd; i++) {
        if (!self->value_in_axis(q[i], i)) {
            for (size_t j = 0; j < nd; j++){
                out[j] = 0;
            }
            return;
        }
    }
    self->interp(out, q);
}

} // namespace ode

#endif // SAMPLED_VECTORFIELDS_IMPL_HPP