#ifndef GRID_INTERP_HPP
#define GRID_INTERP_HPP

#include <cmath>
#include "../OdeInt.hpp"



/**

Experimental header file for regular grid interpolation and vector field streamlines.

*/


namespace ode {

template<typename T, size_t N>
class RegularGridInterpolator{

    /**
    Represents an N-dimensional regular grid interpolator using multilinear interpolation.
    The grid points along each dimension are provided as 1D arrays, and the field values
    are provided as an N-dimensional array.

    Only multilinear interpolation is supported at the moment.
    */

    static_assert(N >= 1, "Number of dimensions must be at least 1");

public:

    template<typename... ViewType>
    RegularGridInterpolator(const T* field_data, const ViewType&... x) : field_(field_data, x.size()...){
        static_assert(sizeof...(x) == N, "Number of grid dimensions must match template parameter N");
        FOR_LOOP(size_t, I, N,
            x_[I] = Array1D<T>(x.data(), x.size());
        );
    }

    inline constexpr size_t ndim() const{
        return N;
    }

    inline T length(size_t axis) const{
        const Array1D<T>& x_grid = x_[axis];
        return x_grid[x_grid.size()-1] - x_grid[0];
    }

    inline const T& x(size_t axis, size_t index) const{
        assert(axis < N && "Axis out of bounds");
        return x_[axis][index];
    }

    inline bool in_bounds(const T& x, size_t axis) const{
        const Array1D<T>& x_grid = x_[axis];
        return (x >= x_grid[0] && x <= x_grid[x_grid.size()-1]);
    }

    inline const T& value(size_t i, size_t j) const{
        return field_(i, j);
    }

    size_t constexpr get_left_nbr(const T& x, size_t axis) const{
        //performs binary search to find the left neighbor index
        assert(in_bounds(x, axis) && "Point out of bounds");
        const T* y = x_[axis].data();
        size_t n = x_[axis].size();
        size_t left = 0;
        size_t right = n - 1;
        while (left < right){
            size_t mid = left + (right - left) / 2;
            if (y[mid] <= x){
                left = mid + 1;
            }else {
                right = mid;
            }
        }
        return left - 1;
    }

    
    T get(const T* q) const{

        Array<T, ndspan::Allocation::Heap, ndspan::Layout::C, N, 2> cube(N, 2); // cube(axis, neighbor)

        std::array<T, N> coefs;

        for (size_t axis=0; axis<N; axis++){
            size_t left_nbr = this->get_left_nbr(q[axis], axis);
            cube(axis, 0) = this->x_[axis][left_nbr];
            cube(axis, 1) = this->x_[axis][left_nbr+1];
            coefs[axis] = (q[axis] - cube(axis, 0))/(cube(axis, 1) - cube(axis, 0));
        }
        

        constexpr size_t n_corners = 1 << N; // 2^N
        T result = 0;
        for (size_t corner = 0; corner < n_corners; corner++){
            T weight = 1;
            size_t offset = 0;
            for (size_t axis=0; axis<N; axis++){
                size_t bit = (corner >> axis) & 1;
                weight *= (bit == 1) ? coefs[axis] : (1 - coefs[axis]);
                offset = offset * x_[axis].size() + (bit == 1 ? this->get_left_nbr(q[axis], axis) + 1 : this->get_left_nbr(q[axis], axis));
            }
            result += weight * field_[offset];
        }
        return result;
    }

    inline const Array1D<T>& x(size_t axis) const{
        assert(axis < N && "Axis out of bounds");
        return x_[axis];
    }

    inline const ndspan::NdArray<T, N>& field() const{
        return field_;
    }

    template<typename... Scalar>
    T operator()(const Scalar&... x) const{
        static_assert(sizeof...(x) == N && "Number of arguments must match number of dimensions");
        // T coords[ndim()] = {static_cast<T>(x)...};
        std::array<T, N> coords = {T(x)...};
        return this->get(coords.data());
    }

private:

    ndspan::NdArray<T, N> field_;
    std::array<Array1D<T>, N> x_;

};



template<typename T>
class SampledVectorField2D{

public:

    SampledVectorField2D(const T* x, const T* y, const T* u, const T* v, size_t Nx, size_t Ny) : u_interp_(u, View1D<T>(x, Nx), View1D<T>(y, Ny)), v_interp_(v, View1D<T>(x, Nx), View1D<T>(y, Ny)) {}
    
    inline bool in_bounds(const T& x, size_t axis) const{
        return u_interp_.in_bounds(x, axis);
    }

    inline const Array1D<T>& x() const{
        return u_interp_.x(0);
    }

    inline const Array1D<T>& y() const{
        return u_interp_.x(1);
    }

    inline const ndspan::NdArray<T, 2>& u_field() const{
        return u_interp_.field();
    }

    inline const ndspan::NdArray<T, 2>& v_field() const{
        return v_interp_.field();
    }

    inline T Lx() const{
        return u_interp_.length(0);
    }

    inline T Ly() const{
        return u_interp_.length(1);
    }

    void get(T* out, const T& x, const T& y) const{
        out[0] = u_interp_(x, y);
        out[1] = v_interp_(x, y);
    }

    void get_norm(T* out, const T& x, const T& y) const{
        T u = u_interp_(x, y);
        T v = v_interp_(x, y);
        T norm = sqrt(u*u + v*v);
        out[0] = u / norm;
        out[1] = v / norm;
    }

    auto ode_func_norm() const {
        return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
            if (!this->in_bounds(q[0], 0) || !this->in_bounds(q[1], 1)){
                out[0] = out[1] = 0;
                return;
            }
            this->get_norm(out, q[0], q[1]);
        };
    }

    auto ode_func() const {
        return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
            if (!this->in_bounds(q[0], 0) || !this->in_bounds(q[1], 1)){
                out[0] = out[1] = 0;
                return;
            }
            this->get(out, q[0], q[1]);
        };
    }

    OdeResult<T> streamline(T x0, T y0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method) const{
        /*
        direction: +1 forward, -1 backward
        */

        std::array<T, 2> y0_vec = {x0, y0};
        ODE<T> ode(OdeData{.rhs=this->ode_func(), .obj=this}, 0, y0_vec.data(), 2, rtol, atol, min_step, max_step, stepsize, direction, {}, {}, method);

        return ode.integrate(length, t_eval);
    }


    std::vector<Array2D<T, 2, 0>> streamplot_data(const T max_length, const T ds, size_t density) const{

        assert(max_length > ds && max_length > 0 && ds > 0 && "max_length and ds must be positive, and max_length must be greater than ds");

        auto ode = this->ode_func_norm();

        using Solver = RK4<T, 2, SolverPolicy::Static, decltype(ode), std::nullptr_t>;

        size_t  Nx = density;
        size_t  Ny = density;
        T       Lx = this->Lx();
        T       Ly = this->Ly();
        T       Dx = Lx / (Nx - 1);
        T       Dy = Ly / (Ny - 1);
        const T* X = this->x().data();
        const T* Y = this->y().data();

        size_t max_pts = size_t(max_length / ds) + 1; //per integration direction

        Array2D<bool> is_full(Nx, Ny);
        std::vector<int> altered_x;
        std::vector<int> altered_y;
        is_full.set(false);

        T min_length = 0.2 * std::min(Lx, Ly);        

        auto GetIdx = [&](size_t& i, size_t& j, const T& x, const T& y) LAMBDA_INLINE{
            i = size_t(std::round((x - X[0]) / Dx));
            j = size_t(std::round((y - Y[0]) / Dy));
        };

        auto in_bounds = [&](int i, int j) LAMBDA_INLINE{
            return (i >= 0 && i < int(Nx) && j >= 0 && j < int(Ny));
        };

        auto IntegrateDirection = [&](Solver& solver, T& s_total, T* x, T* y, size_t& n_steps_tot, const int dir){
            // x and y are preallocated arrays of size max_pts + 1. x[0] = x0, y[0] = y0, so at each step, we write to x[step], y[step], starting from x[1], y[1]
            const T& x0 = x[0];
            const T& y0 = y[0];
            assert(this->in_bounds(x0, 0) && this->in_bounds(y0, 1) && "Initial point out of bounds");
            assert(dir == 1 || dir == -1 && "Direction must be either +1 or -1");
            size_t n_steps = 0;
            std::array<T, 2> ics = {x0, y0};
            solver.set_ics(0, ics.data(), ds, dir);
            const T* q_new;
            const T* q_old;
            size_t i_start, j_start;
            GetIdx(i_start, j_start, x0, y0);
            size_t i, j;
            while ((n_steps < max_pts) && solver.advance()){
                q_new = solver.vector().data();
                q_old = solver.vector_old().data();
                GetIdx(i, j, q_new[0], q_new[1]);
                s_total += sqrt((q_new[0] - q_old[0])*(q_new[0] - q_old[0]) + (q_new[1] - q_old[1])*(q_new[1] - q_old[1]));
                n_steps++;
                n_steps_tot++;
                x[n_steps*dir] = q_new[0];
                y[n_steps*dir] = q_new[1];
                if ((i != i_start || j != j_start) && in_bounds(i, j)){
                    if(!is_full(i, j)){
                        is_full(i, j) = true;
                        altered_x.push_back(i);
                        altered_y.push_back(j);
                        i_start = i;
                        j_start = j;
                    }else {
                        break;
                    }
                }
            }
            return n_steps;
        };


        auto GetStreamline = [&](bool& success, Array2D<T, 2, 0>& worker_line, T x0, T y0) {
            assert(worker_line.shape(0) == 2 && worker_line.shape(1) == 2*max_pts + 1 && "Line array must have shape (2, 2*max_pts + 1)");
            worker_line(0, max_pts) = x0;
            worker_line(1, max_pts) = y0;

            size_t n_steps_tot = 0;
            Solver solver({.rhs=ode}, 0, nullptr, 2, 1e-5, 1e-5, 0, inf<T>(), ds, +1);

            T s_total = 0;
            altered_x.clear();
            altered_y.clear();
            
            IntegrateDirection(solver, s_total, worker_line.ptr(0, max_pts), worker_line.ptr(1, max_pts), n_steps_tot, +1);
            size_t n_steps_bwd = IntegrateDirection(solver, s_total, worker_line.ptr(0, max_pts), worker_line.ptr(1, max_pts), n_steps_tot, -1);

            // if the streamline is long enough and n_steps_tot  > 1, we keep it:
            const T* x_line = worker_line.ptr(0, max_pts - n_steps_bwd);
            const T* y_line = worker_line.ptr(1, max_pts - n_steps_bwd);
            Array2D<T, 2, 0> true_line(2, n_steps_tot);
            if (s_total > min_length && n_steps_tot > 1){
                copy_array(true_line.data(), worker_line.ptr(0, max_pts - n_steps_bwd), n_steps_tot);
                copy_array(true_line.data() + n_steps_tot, worker_line.ptr(1, max_pts - n_steps_bwd), n_steps_tot);
                
                //mark the initial point
                size_t i_start, j_start;
                GetIdx(i_start, j_start, x0, y0);
                if (in_bounds(i_start, j_start)){
                    is_full(i_start, j_start) = true;
                }
                success = true;
            }else{
                // unmark the points we marked during this integration attempt
                for (size_t k = 0; k < altered_x.size(); k++){
                    is_full(altered_x[k], altered_y[k]) = false;
                }
                success = false;
            }
            return true_line;
        };

        std::vector<Array2D<T, 2, 0>> streamlines;
        auto TryTrajectory = [&](Array2D<T, 2, 0>& worker_line, int i, int j) LAMBDA_INLINE{
            if (i < 0 || i >= Nx || j < 0 || j >= Ny){
                return;
            }
            T x0 = X[0] + i * Dx;
            T y0 = Y[0] + j * Dy;
            assert(this->in_bounds(x0, 0) && this->in_bounds(y0, 1) && "Initial point out of bounds");
            if (!is_full(i, j)){
                bool success;
                auto new_line = GetStreamline(success, worker_line, x0, y0);
                if (success) {
                    streamlines.push_back(std::move(new_line));
                }
            }
        };

        // Seeding streamlines from edges working inwards in a spiral pattern:
        Array2D<T, 2, 0> line(2, 2*max_pts + 1);
        for (size_t indent = 0; indent < std::max(Nx, Ny) / 2; indent++){
            for (size_t i = 0; i < std::max(Nx, Ny) - 2*indent; i++){
                TryTrajectory(line, indent + i, indent);
                TryTrajectory(line, indent + i, Ny - 1 - indent);
                TryTrajectory(line, indent, indent + i);
                TryTrajectory(line, Nx - 1 - indent, indent + i);
            }
        }
        return streamlines;
    }

private:
    RegularGridInterpolator<T, 2> u_interp_;
    RegularGridInterpolator<T, 2> v_interp_;
};


} // namespace ode

#endif // GRID_INTERP_HPP