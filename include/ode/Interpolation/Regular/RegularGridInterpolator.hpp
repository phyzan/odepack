#ifndef REGULAR_GRID_INTERPOLATOR_HPP
#define REGULAR_GRID_INTERPOLATOR_HPP

#include "Grids.hpp"
#include "../NdInterpolator.hpp"
#include "../VectorFields.hpp"

namespace ode {



template<typename T, int NDIM, bool AS_VIRTUAL = false>
class RegularGridInterpolator : public NdInterpolator<RegularGridInterpolator<T, NDIM, AS_VIRTUAL>, T, NDIM, AS_VIRTUAL>{

    /**
    Represents an N-dimensional regular grid interpolator using multilinear interpolation.
    The grid points along each dimension are provided as 1D arrays, and the field values
    are provided as an N-dimensional array.

    Only multilinear interpolation is supported at the moment.
    */

    using Base = NdInterpolator<RegularGridInterpolator<T, NDIM, AS_VIRTUAL>, T, NDIM, AS_VIRTUAL>;

public:

    // values must be a contiguous array of shape (n_points, nfields) if coord_axis_first, or (nfields, n_points) if not
    // where the n_points part can be reshaped to (nx, ny, ...) as a C-contiguous array
    template<typename ValuesContainer, typename AxisViewContainer>
    RegularGridInterpolator(const ValuesContainer& values, const AxisViewContainer& grid, bool coord_axis_first);

    inline const RegularGrid<T, NDIM>&  grid() const { return grid_; }

    // ============== Static Override ==================
    int             ndim() const;
    bool            interp(T* out, const T* coords) const;
    bool            contains(const T* coords) const;
    // =================================================

private:

    RegularGrid<T, NDIM> grid_;

}; // RegularGridInterpolator


template<typename T, int NDIM, bool AS_VIRTUAL = false>
class RegularVectorField : public RegularGridInterpolator<T, NDIM, AS_VIRTUAL>, public VectorField<RegularVectorField<T, NDIM, AS_VIRTUAL>, T, NDIM, AS_VIRTUAL>{

    using InterpBase = RegularGridInterpolator<T, NDIM, AS_VIRTUAL>;
    using VFBase = VectorField<RegularVectorField<T, NDIM, AS_VIRTUAL>, T, NDIM, AS_VIRTUAL>;

public:

    // grid[i].data(), grid[i].size() : grid points along axis i
    template<typename ValuesContainer, typename AxisViewContainer>
    RegularVectorField(const ValuesContainer& values, const AxisViewContainer& grid, bool coord_axis_first);

    std::vector<Array2D<T, NDIM, 0>>    streamplot_data(T max_length, T ds, size_t density, T rtol, T atol, T min_step, T max_step, T stepsize, Integrator method) const;


    // ============ Explicit overrides for VectorField ==============
    bool interp(T* out, const T* coords) const;
    int ndim() const;
    bool contains(const T* coords) const;
    // ==============================================================


private:

    template<size_t... I>
    std::vector<Array2D<T, NDIM, 0>>    streamplot_data_core(T max_length, T ds, size_t density, T rtol, T atol, T min_step, T max_step, T stepsize, Integrator method, std::index_sequence<I...>) const;

}; // RegularVectorField



template<typename AxisViewContainer>
inline int get_point_count(const AxisViewContainer& grid){
    int count = 1;
    for (size_t i=0; i<grid.size(); i++){
        count *= grid[i].size();
    }
    return count;
}

} // namespace ode

#endif // REGULAR_GRID_INTERPOLATOR_HPP
