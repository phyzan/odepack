#ifndef GRID_INTERP_HPP
#define GRID_INTERP_HPP


#include "ode.hpp"



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

    OdeResult<T> streamline(T x0, T y0, T length, T rtol, T atol, T min_step, T max_step, T first_step, int direction, const StepSequence<T>& t_eval, const std::string& method) const{
        /*
        direction: +1 forward, -1 backward
        */
        auto ode_func = [](T* out, const T& t, const T* q, const T* args, const void* ptr){
            const auto* vec_field = reinterpret_cast<const SampledVectorField2D<T>*>(ptr);
            if (!vec_field->in_bounds(q[0], 0) || !vec_field->in_bounds(q[1], 1)){
                out[0] = out[1] = 0;
                return;
            }
            vec_field->get_norm(out, q[0], q[1]);
        };


        std::array<T, 2> y0_vec = {x0, y0};
        ODE<T> ode({.rhs=ode_func, .obj=this}, 0, y0_vec.data(), 2, rtol, atol, min_step, max_step, first_step, direction, {}, {}, method);

        return ode.integrate(length, t_eval);
    }

private:
    RegularGridInterpolator<T, 2> u_interp_;
    RegularGridInterpolator<T, 2> v_interp_;
};


} // namespace ode

#endif // GRID_INTERP_HPP