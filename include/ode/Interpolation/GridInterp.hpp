#ifndef GRID_INTERP_HPP
#define GRID_INTERP_HPP

#include <cmath>
#include "../OdeInt.hpp"



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

    inline constexpr size_t                     ndim() const;

    inline constexpr size_t                     nfields() const;

    inline T                                    length(size_t axis) const;

    inline const std::array<Array1D<T>, NDIM>&  x_all() const;

    inline const Array1D<T>&                    x_data(size_t axis) const;

    template<typename... Scalar>
    inline bool                                 coords_in_bounds(const Scalar&... x) const;

    inline bool                                 value_in_axis(const T& x, size_t axis) const;

    template<INT_T... Int>
    inline const T&                             value(size_t field, Int... idx) const;

    size_t constexpr                            get_left_nbr(const T& x, size_t axis) const;

    inline const Array1D<T>&                    x(size_t axis) const;

    inline const NdArray<T, NDIM>&              field(size_t idx) const;

    inline void                                 get(T* out, const T* q) const;

    template<size_t FieldIdx>
    inline T                                    get_single(const T* q) const;

    template<typename... Scalar>
    void                                        fill(T* out, const Scalar&... x) const;

    void                                        get_norm(T* out, const T* q) const;

    template<typename... Scalar>
    void                                        fill_norm(T* out, const Scalar&... x) const;

private:

    template<int FieldIdx = -1>
    void get_helper(T* out, const T* q) const;

    std::vector<NdArray<T, NDIM>> field_;
    std::array<Array1D<T>, NDIM> x_;

};




// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

#ifndef NO_ODE_TEMPLATE

// ----------------------------------------------------------------------------
// RegularGridInterpolator
// ----------------------------------------------------------------------------

template<typename T, size_t NDIM>
template<typename... Args>
RegularGridInterpolator<T, NDIM>::RegularGridInterpolator(const Args&... args){
    static_assert(sizeof...(args) > NDIM, "Number of arguments must match number of fields plus number of dimensions");

    constexpr size_t NFIELD = sizeof...(args) - NDIM;
    field_.resize(NFIELD);

    FOR_LOOP(size_t, I_X, NDIM,
        const auto& x_I = pack_elem<I_X>(args...);
        x_[I_X] = Array1D<T>(x_I.data(), x_I.size());
    );

    FOR_LOOP(size_t, I_FIELD, NFIELD,
        const auto& field_I = pack_elem<NDIM + I_FIELD>(args...);
        EXPAND(size_t, NDIM, I_X,
            field_[I_FIELD] = NdArray<T, NDIM>(field_I, pack_elem<I_X>(args...).size()...);
        );
    );
}

#endif // NO_ODE_TEMPLATE

template<typename T, size_t NDIM>
inline constexpr size_t RegularGridInterpolator<T, NDIM>::ndim() const{
    return NDIM;
}

template<typename T, size_t NDIM>
inline constexpr size_t RegularGridInterpolator<T, NDIM>::nfields() const{
    return field_.size();
}

template<typename T, size_t NDIM>
inline T RegularGridInterpolator<T, NDIM>::length(size_t axis) const{
    const Array1D<T>& x_grid = x_[axis];
    return x_grid[x_grid.size()-1] - x_grid[0];
}

template<typename T, size_t NDIM>
inline const std::array<Array1D<T>, NDIM>& RegularGridInterpolator<T, NDIM>::x_all() const{
    return x_;
}

template<typename T, size_t NDIM>
inline const Array1D<T>& RegularGridInterpolator<T, NDIM>::x_data(size_t axis) const{
    assert(axis < NDIM && "Axis out of bounds");
    return x_[axis];
}

template<typename T, size_t NDIM>
template<typename... Scalar>
inline bool RegularGridInterpolator<T, NDIM>::coords_in_bounds(const Scalar&... x) const{
    static_assert(sizeof...(x) == NDIM, "Number of arguments must match number of dimensions");
    return EXPAND(size_t, NDIM, I,
        return ((x >= x_[I][0] && x <= x_[I][x_[I].size()-1]) && ...);
    );
}

template<typename T, size_t NDIM>
inline bool RegularGridInterpolator<T, NDIM>::value_in_axis(const T& x, size_t axis) const{
    assert(axis < NDIM && "Axis out of bounds");
    const Array1D<T>& X = x_[axis];
    return (x >= X[0] && x <= X[X.size()-1]);
}

template<typename T, size_t NDIM>
template<INT_T... Int>
inline const T& RegularGridInterpolator<T, NDIM>::value(size_t field, Int... idx) const{
    assert(field < this->nfields() && "Field index out of bounds");
    return field_[field](idx...);
}

template<typename T, size_t NDIM>
size_t constexpr RegularGridInterpolator<T, NDIM>::get_left_nbr(const T& x, size_t axis) const{
    //performs binary search to find the left neighbor index
    assert(value_in_axis(x, axis) && "Point out of bounds");
    const Array1D<T>& X = x_[axis];
    size_t n = X.size();
    size_t left = 0;
    size_t right = n - 1;
    while (left < right){
        size_t mid = left + (right - left) / 2;
        if (X[mid] <= x){
            left = mid + 1;
        }else {
            right = mid;
        }
    }
    return left - 1;
}

template<typename T, size_t NDIM>
inline const Array1D<T>& RegularGridInterpolator<T, NDIM>::x(size_t axis) const{
    assert(axis < NDIM && "Axis out of bounds");
    return x_[axis];
}

template<typename T, size_t NDIM>
inline const NdArray<T, NDIM>& RegularGridInterpolator<T, NDIM>::field(size_t idx) const{
    assert(idx < this->nfields() && "Field index out of bounds");
    return field_[idx];
}

template<typename T, size_t NDIM>
inline void RegularGridInterpolator<T, NDIM>::get(T* out, const T* q) const{
    get_helper<-1>(out, q);
}

template<typename T, size_t NDIM>
template<size_t FieldIdx>
inline T RegularGridInterpolator<T, NDIM>::get_single(const T* q) const{
    assert(FieldIdx < this->nfields() && "FieldIdx template parameter out of bounds");
    T out;
    get_helper<FieldIdx>(&out, q);
    return out;
}

template<typename T, size_t NDIM>
template<typename... Scalar>
void RegularGridInterpolator<T, NDIM>::fill(T* out, const Scalar&... x) const{
    static_assert(sizeof...(x) == NDIM && "Number of arguments must match number of dimensions");
    // T coords[ndim()] = {static_cast<T>(x)...};
    std::array<T, NDIM> coords = {T(x)...};
    this->get(out, coords.data());
}

template<typename T, size_t NDIM>
void RegularGridInterpolator<T, NDIM>::get_norm(T* out, const T* q) const{
    this->get(out, q);
    T norm = 0;
    for (size_t i=0; i<this->nfields(); i++){
        norm += out[i]*out[i];
    }
    norm = sqrt(norm);
    for (size_t i=0; i<this->nfields(); i++){
        out[i] /= norm;
    }
}

template<typename T, size_t NDIM>
template<typename... Scalar>
void RegularGridInterpolator<T, NDIM>::fill_norm(T* out, const Scalar&... x) const{
    static_assert(sizeof...(x) == NDIM && "Number of arguments must match number of dimensions");
    std::array<T, NDIM> coords = {T(x)...};
    this->get_norm(out, coords.data());
}

template<typename T, size_t NDIM>
template<int FieldIdx>
void RegularGridInterpolator<T, NDIM>::get_helper(T* out, const T* q) const{
    // If FieldIdx = -1, then we fill the entire output array.
    // Otherwise, we only fill *out with the field corresponding to FieldIdx, and ignore the rest of the output array.
    static_assert(FieldIdx >= -1, "FieldIdx template parameter must be -1 or a non-negative integer");
    assert((FieldIdx == -1 || FieldIdx < this->nfields()) && "FieldIdx template parameter out of bounds");
    Array<T, Allocation::Heap, Layout::C, NDIM, 2> cube(NDIM, 2); // cube(axis, neighbor)

    std::array<T, NDIM> coefs;

    for (size_t axis=0; axis<NDIM; axis++){
        size_t left_nbr = this->get_left_nbr(q[axis], axis);
        cube(axis, 0) = this->x_[axis][left_nbr];
        cube(axis, 1) = this->x_[axis][left_nbr+1];
        coefs[axis] = (q[axis] - cube(axis, 0))/(cube(axis, 1) - cube(axis, 0));
    }


    constexpr size_t n_corners = 1 << NDIM; // 2^NDIM
    // initialize output to 0
    if constexpr (FieldIdx != -1){
        *out = 0;
    } else {
        for (size_t field=0; field<this->nfields(); field++){
            out[field] = 0;
        }
    }

    // perform multilinear interpolation
    for (size_t corner = 0; corner < n_corners; corner++){
        T weight = 1;
        size_t offset = 0;
        for (size_t axis=0; axis<NDIM; axis++){
            size_t bit = (corner >> axis) & 1;
            weight *= (bit == 1) ? coefs[axis] : (1 - coefs[axis]);
            offset = offset * x_[axis].size() + (bit == 1 ? this->get_left_nbr(q[axis], axis) + 1 : this->get_left_nbr(q[axis], axis));
        }
        if constexpr (FieldIdx != -1){
            *out += weight * field_[FieldIdx][offset];
        } else {
            for (size_t field=0; field<this->nfields(); field++){
                out[field] += weight * field_[field][offset];
            }
        }
    }
}

} // namespace ode

#endif // GRID_INTERP_HPP
