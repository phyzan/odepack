#ifndef GRID_INTERP_HPP
#define GRID_INTERP_HPP

#include <cmath>
#include "../../ndspan.hpp"



namespace ode {

using namespace ndspan;

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T, size_t NDIM>
class RegularGridInterpolator{

    /**
    Represents an N-dimensional regular grid interpolator using multilinear interpolation.
    The grid points along each dimension are provided as 1D arrays, and the field values
    are provided as an N-dimensional array.

    Only multilinear interpolation is supported at the moment.
    */

    static_assert(NDIM >= 1, "Number of dimensions and fields must be at least 1");

public:

    // The constructor must be called like f(x1_grid, ..., xN_grid, field_1_data, ..., field_NFIELD_data), where x1_grid, ..., xN_grid are 1D arrays representing the grid points along each dimension, and field_1_data, ..., field_NFIELD_data are N-dimensional arrays representing the field values at the grid points. The number of grid arrays must match NDIM, and the number of field arrays must match NFIELD.
    template<typename... Args>
    RegularGridInterpolator(const Args&... args);

    constexpr size_t                    ndim() const;

    constexpr size_t                    nfields() const;

    T                                   length(size_t axis) const;

    const std::array<Array1D<T>, NDIM>& x_all() const;

    const Array1D<T>&                   x_data(size_t axis) const;

    template<typename... Scalar>
    bool                                coords_in_bounds(const Scalar&... x) const;

    bool                                value_in_axis(const T& x, size_t axis) const;

    template<INT_T... Int>
    const T&                            value(size_t field, Int... idx) const;

    size_t constexpr                    get_left_nbr(const T& x, size_t axis) const;

    const Array1D<T>&                   x(size_t axis) const;

    const NdArray<T, NDIM>&             field(size_t idx) const;

    void                                get(T* out, const T* q) const;

    template<size_t FieldIdx>
    T                                   get_single(const T* q) const;

    template<typename... Scalar>
    void                                fill(T* out, const Scalar&... x) const;

    void                                get_norm(T* out, const T* q) const;

    template<typename... Scalar>
    void                                fill_norm(T* out, const Scalar&... x) const;

private:

    template<int FieldIdx = -1>
    void get_helper(T* out, const T* q) const;

    std::vector<NdArray<T, NDIM>> field_;
    std::array<Array1D<T>, NDIM> x_;

};

} // namespace ode

#endif // GRID_INTERP_HPP
