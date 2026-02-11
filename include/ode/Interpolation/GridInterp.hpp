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

public:

    static constexpr size_t FDIM = (NDIM == 0 ? 0 : NDIM + 1);
    static constexpr Allocation Alloc = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);

    RegularGridInterpolator(const T* values, size_t Nvals, std::vector<View1D<T>> grid, bool values_axis_first = false);

    // The ownership of grid MutView in the grid will be transferred to the interpolator.
    // The user must not use the grid MutView after passing it to this constructor,
    // and they must hold the memory address of dynamically allocated arrays
    RegularGridInterpolator(const T* values, size_t Nvals, std::vector<MutView1D<T>> grid, bool values_axis_first = false);

    constexpr size_t                    ndim() const;

    inline const NdArray<T, FDIM>&      values() const { return field_; }

    T                                   length(size_t axis) const;

    const Array1D<Array1D<T>, NDIM, Alloc>& x_all() const;

    const Array1D<T>&                   x_data(size_t axis) const;

    bool                                coords_in_bounds(const T* coords) const;

    bool                                value_in_axis(const T& x, size_t axis) const;

    size_t constexpr                    get_left_nbr(const T& x, size_t axis) const;

    const Array1D<T>&                   x(size_t axis) const;

    void                                interp(T* out, const T* q) const;

    void                                interp_norm(T* out, const T* q) const;

private:

    NdArray<T, FDIM> field_; // shape = (nx, ny, ..., nfields)
    Array1D<Array1D<T>, NDIM, Alloc> x_;

};

} // namespace ode

#endif // GRID_INTERP_HPP
