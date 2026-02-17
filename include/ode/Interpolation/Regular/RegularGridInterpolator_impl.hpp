#ifndef REGULAR_GRID_INTERPOLATOR_IMPL_HPP
#define REGULAR_GRID_INTERPOLATOR_IMPL_HPP

#include "RegularGridInterpolator.hpp"

namespace ode{


// ============================================================================================
//                              RegularGridInterpolator
// ============================================================================================

template<typename T, int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer, typename AxisViewContainer>
RegularGridInterpolator<T, NDIM, AS_VIRTUAL>::RegularGridInterpolator(const ValuesContainer& values, const AxisViewContainer& grid, bool coord_axis_first) : Base(values, coord_axis_first), grid_(grid) {
    assert([&](){
        int tot_size = coord_axis_first ? values.shape(0) : values.shape(values.ndim()-1);
        return static_cast<bool>(tot_size == get_point_count(grid));
    }() && "Mismatch between values and grid points");
}

template<typename T, int NDIM, bool AS_VIRTUAL>
bool RegularGridInterpolator<T, NDIM, AS_VIRTUAL>::contains(const T* coords) const{
    return grid_.contains(coords);
}

template<typename T, int NDIM, bool AS_VIRTUAL>
int RegularGridInterpolator<T, NDIM, AS_VIRTUAL>::ndim() const{
    return grid_.ndim();
}


template<typename T, int NDIM, bool AS_VIRTUAL>
bool RegularGridInterpolator<T, NDIM, AS_VIRTUAL>::interp(T* out, const T* coords) const{
    int nd = this->ndim();
    Array2D<T, NDIM, 2, Base::ALLOC> cube(nd, 2); // cube(axis, neighbor)

    Array1D<T, NDIM, Base::ALLOC> coefs(nd); // interpolation coefficients for each axis
    Array1D<int, NDIM, Base::ALLOC> left_nbrs(nd); // left neighbors for each axis
    int nfields = this->nvals_per_point();
    const T* values = this->values().data();

    const RegularGrid<T, NDIM>& grid = this->grid();

    for (int axis=0; axis<nd; axis++){
        int left_nbr = grid.find_left_idx(coords[axis], axis);
        if (left_nbr == -1){
            return false; // point is out of bounds along this axis
        }
        left_nbrs[axis] = left_nbr;
        cube(axis, 0) = grid.x(axis)[left_nbr];
        cube(axis, 1) = grid.x(axis)[left_nbr+1];
        coefs[axis] = (coords[axis] - cube(axis, 0))/(cube(axis, 1) - cube(axis, 0));
    }


    int n_corners = 1 << nd; // 2^ndim
    // initialize output to 0
    for (int field=0; field<nfields; field++){
        out[field] = 0;
    }

    // perform multilinear interpolation
    for (int corner = 0; corner < n_corners; corner++){
        T weight = 1;
        int offset = 0;
        for (int axis=0; axis<nd; axis++){
            int bit = (corner >> axis) & 1;
            weight *= (bit == 1) ? coefs[axis] : (1 - coefs[axis]);
            offset = offset * grid.x(axis).size() + (bit == 1 ? left_nbrs[axis] + 1 : left_nbrs[axis]);
        }
        for (int field=0; field<nfields; field++){
            out[field] += weight * values[offset*nfields + field];
        }
    }
    return true;
}

// ============================================================================================
//                              RegularVectorField
// ============================================================================================

template<typename T, int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer, typename AxisViewContainer>
RegularVectorField<T, NDIM, AS_VIRTUAL>::RegularVectorField(const ValuesContainer& values, const AxisViewContainer& grid, bool coord_axis_first) : InterpBase(values, grid, coord_axis_first), VFBase() {}

template<typename T, int NDIM, bool AS_VIRTUAL>
bool RegularVectorField<T, NDIM, AS_VIRTUAL>::interp(T* out, const T* coords) const{
    return InterpBase::interp(out, coords);
}

template<typename T, int NDIM, bool AS_VIRTUAL>
int RegularVectorField<T, NDIM, AS_VIRTUAL>::ndim() const{
    return InterpBase::ndim();
}

template<typename T, int NDIM, bool AS_VIRTUAL>
bool RegularVectorField<T, NDIM, AS_VIRTUAL>::contains(const T* coords) const{
    return InterpBase::contains(coords);
}


template<typename T, int NDIM, bool AS_VIRTUAL>
std::vector<Array2D<T, NDIM, 0>> RegularVectorField<T, NDIM, AS_VIRTUAL>::streamplot_data(const T& max_length, const T& ds, size_t density, double rtol, double atol, double min_step, double max_step, const std::string& method) const{
    return streamplot_data_core(max_length, ds, density, rtol, atol, min_step, max_step, method, std::make_index_sequence<NDIM>{});
}



template<typename T, int NDIM, bool AS_VIRTUAL>
template<size_t... I>
std::vector<Array2D<T, NDIM, 0>> RegularVectorField<T, NDIM, AS_VIRTUAL>::streamplot_data_core(const T& max_length, const T& ds, size_t density, double rtol, double atol, double min_step, double max_step, const std::string& method, std::index_sequence<I...>) const{
    
    assert(max_length > ds && max_length > 0 && ds > 0 && "max_length and ds must be positive, and max_length must be greater than ds");
    assert(density > 1 && "Density must be greater than 1");

    // using RK4<T, 0, RichVirtual> because it is the Python bindings instanciate these templates parameters,
    // event though they are not the optimal ones. Ideally we want RK4<T, NDIM, Static, decltype(lambda_func), std::nullptr_t>, but the performance
    // gain is far too small to justify the increase in binary size and compilation time.

    std::vector<T> args{};
    std::vector<const Event<T>*> events{};
    std::unique_ptr<OdeRichSolver<T, NDIM>> unique_solver = get_virtual_solver<T, NDIM>(method, OdeData<Func<T>, void>{.rhs=VFBase::ode_func_norm, .obj=this}, 0, nullptr, this->ndim(), rtol, atol, min_step, max_step, ds, +1, static_cast<const std::vector<T>&>(args), static_cast<const std::vector<const Event<T>*>&>(events));
    OdeRichSolver<T, NDIM>* solver = unique_solver.get();

    const auto& X = this->grid().data();
    Array1D<int, NDIM, InterpBase::ALLOC> N(this->ndim());
    Array1D<T, NDIM, InterpBase::ALLOC> L(this->ndim());
    Array1D<T, NDIM, InterpBase::ALLOC> Dx(this->ndim());
    for (int axis=0; axis<this->ndim(); axis++){
        N[axis] = density;
        L[axis] = this->grid().length(axis);
        Dx[axis] = L[axis] / (N[axis] - 1);
    }


    int max_pts = int(max_length / ds) + 1; //per integration direction

    NdArray<bool, NDIM> is_full(nullptr, N.data(), this->ndim());
    Array1D<std::vector<int>, NDIM, InterpBase::ALLOC> reached(this->ndim()); // reached[axis] = vector of reached indices along that axis
    is_full.set(false);

    T min_length = 0.2 * (*std::min_element(L.begin(), L.end())); // minimum length of streamline to be kept, as a fraction of the minimum grid length

    Array1D<int, NDIM, InterpBase::ALLOC> i_start(this->ndim());
    Array1D<int, NDIM, InterpBase::ALLOC> i_curr(this->ndim());

    auto GetIdx = [&](Array1D<int, NDIM, InterpBase::ALLOC>& i, const T* x) LAMBDA_INLINE{
        if constexpr (NDIM > 0){
            ((i[I] = int(std::round((x[I] - X[I][0]) / Dx[I]))), ...);
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                i[axis] = int(std::round((x[axis] - X[axis][0]) / Dx[axis]));
            }
        }
    };

    Array1D<int, NDIM, InterpBase::ALLOC> idx_aux(this->ndim());

    auto InBounds = [&](const Array1D<int, NDIM, InterpBase::ALLOC>& i) LAMBDA_INLINE{
        if constexpr (NDIM > 0){
            return ((i[I] >= 0 && i[I] < int(N[I])) && ...);
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
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
            for (int axis=0; axis<this->ndim(); axis++){
                sum += (q_new[axis] - q_old[axis]) * (q_new[axis] - q_old[axis]);
            }
            return sqrt(sum);
        }
    };

    auto IntegrateDirection = [&](OdeRichSolver<T, NDIM>* solver, T& s_total, T* const * x, int& n_steps_tot, const int dir) -> int {
        // x[I] are preallocated arrays of size max_pts + 1. x[I][0] = x0[I], so at each step, we write to x[I][step], starting from x[I][1]
        Array1D<T, NDIM, InterpBase::ALLOC> ics(this->ndim());
        if constexpr (NDIM > 0){
            ics = Array1D<T, NDIM, InterpBase::ALLOC>{x[I][0]...};
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                ics[axis] = x[axis][0];
            }
        }
        
        assert(this->contains(ics.data()) && "Initial point out of bounds");
        assert((dir == 1 || dir == -1) && "Direction must be either +1 or -1");
        if (!solver->set_ics(0, ics.data(), ds, dir)) {
            return 0;
        }
        int n_steps = 0;
        const T* q_new;
        const T* q_old;


        GetIdx(i_start, ics.data());
        while ((n_steps < max_pts) && solver->advance()){
            q_new = solver->vector().data();
            q_old = solver->vector_old().data();
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
                for (int axis=0; axis<this->ndim(); axis++){
                    x[axis][n_steps*dir] = q_new[axis];
                }
                if ((!allEqual(i_start.data(), i_curr.data(), this->ndim())) && InBounds(i_curr)){
                    if(!is_full.getElem(i_curr.data())){
                        is_full.getElem(i_curr.data()) = true;
                        for (int axis=0; axis<this->ndim(); axis++){
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
        assert(worker_line.shape(1) == size_t(2*max_pts + 1)&& "Line array must have shape (2, 2*max_pts + 1)");
        if constexpr (NDIM > 0) {
            ((worker_line(I, max_pts) = x0[I]), ...);
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                worker_line(axis, max_pts) = x0[axis];
            }
        }
        

        int n_steps_tot = 0;

        // The args and events are passed as const & because this exact signature is compiled in the
        // Python bindings, so this saves binary size and compilation time.

        T s_total = 0;
        Array1D<T*, NDIM, InterpBase::ALLOC> x(this->ndim());
        if constexpr (NDIM > 0){
            x = {worker_line.ptr(I, max_pts)...};
            ((reached[I].clear()), ...);
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                x[axis] = worker_line.ptr(axis, max_pts);
                reached[axis].clear();
            }
        }

        IntegrateDirection(solver, s_total, x.data(), n_steps_tot, +1);
        int n_steps_bwd = IntegrateDirection(solver, s_total, x.data(), n_steps_tot, -1);

        // if the streamline is long enough and n_steps_tot  > 1, we keep it:
        Array1D<const T*, NDIM, InterpBase::ALLOC> x_line(this->ndim());
        if constexpr (NDIM > 0){
            x_line = {worker_line.ptr(I, max_pts - n_steps_bwd)...};
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                x_line[axis] = worker_line.ptr(axis, max_pts - n_steps_bwd);
            }
        }
        Array2D<T, NDIM, 0> true_line(this->ndim(), n_steps_tot);
        if (s_total > min_length && n_steps_tot > 1){
            if constexpr (NDIM > 0){
                ((copy_array(true_line.ptr(I, 0), x_line[I], n_steps_tot)), ...);
            }else{
                for (int axis=0; axis<this->ndim(); axis++){
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
            int r_size = reached[0].size();
            for (int k = 0; k < r_size; k++){
                if constexpr (NDIM > 0){
                    is_full(reached[I][k]...) = false;
                }else{
                    for (int axis=0; axis<this->ndim(); axis++){
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
    auto TryTrajectory = [&](Array2D<T, NDIM, 0>& worker_line, const Array1D<int, NDIM, InterpBase::ALLOC>& idx) LAMBDA_INLINE{

        if constexpr (NDIM > 0) {
            if (((idx[I] < 0 || idx[I] >= int(N[I])) || ...)) {
                return;
            }
        } else{
            for (int axis=0; axis<this->ndim(); axis++){
                if (idx[axis] < 0 || idx[axis] >= int(N[axis])){
                    return;
                }
            }
        }
        Array1D<T, NDIM, InterpBase::ALLOC> x0(this->ndim());
        if constexpr (NDIM > 0){
            x0 = Array1D<T, NDIM, InterpBase::ALLOC>{(X[I][0] + idx[I] * Dx[I])...};
        }else{
            for (int axis=0; axis<this->ndim(); axis++){
                x0[axis] = X[axis][0] + idx[axis] * Dx[axis];
            }
        }

        assert(this->contains(x0.data()) && "Initial point out of bounds");
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

    int min_N = *std::min_element(N.begin(), N.end());
    int max_shells = (min_N + 1) / 2;

    // Compute shell level: minimum distance to any boundary
    auto shell_level = [&N, this](const Array1D<int, NDIM, InterpBase::ALLOC>& idx) -> int {
        int min_dist = std::numeric_limits<int>::max();
        for (int d = 0; d < this->ndim(); d++) {
            int dist = std::min(int(idx[d]), N[d] - 1 - idx[d]);
            min_dist = std::min(min_dist, dist);
        }
        return min_dist;
    };

    for (int shell = 0; shell < max_shells; shell++) {
        // Iterate over all grid points within the current shell's bounding box
        Array1D<int, NDIM, InterpBase::ALLOC> idx(this->ndim());
        std::fill(idx.begin(), idx.end(), shell);
        while (true) {
            // Only process points that are exactly on this shell boundary
            if (shell_level(idx) == shell) {
                TryTrajectory(line, idx);
            }

            // Increment multi-dimensional index
            int d = 0;
            while (d < this->ndim()) {
                idx[d]++;
                if (idx[d] < N[d] - shell) {
                    break;
                }
                idx[d] = shell;
                d++;
            }
            if (d == this->ndim()) {break;} // All combinations exhausted
        }
    }
    return streamlines;
}

} // namespace ode


#endif // REGULAR_GRID_INTERPOLATOR_IMPL_HPP