#ifndef GridInterp_IMPL_HPP
#define GridInterp_IMPL_HPP

#include "GridInterp.hpp"

namespace ode{


// ----------------------------------------------------------------------------
// RegularGridInterpolator
// ----------------------------------------------------------------------------

template<typename T, size_t NDIM>
RegularGridInterpolator<T, NDIM>::RegularGridInterpolator(const T* values, size_t Nvals, std::vector<View1D<T>> grid, bool values_axis_first) : x_(grid.size()){
    size_t nd = grid.size();
    assert(NDIM == 0 || nd == NDIM && "Number of grid arrays must match number of dimensions");
    Array1D<size_t, FDIM, Alloc> v_shape(nd+1);
    for (size_t i=0; i<nd; i++){
        v_shape[i] = grid[i].size();
        x_[i] = Array1D<T>(grid[i].data(), grid[i].size());
    }
    v_shape[nd] = Nvals;
    field_.resize(v_shape.data(), v_shape.size());
    if (!values_axis_first){
        copy_array(field_.data(), values, field_.size());
    }else{
        size_t grid_points = prod(v_shape.data(), nd);
        assert(grid_points * Nvals == field_.size() && "Values array size not conserved, internal bug"); // sanity check
        for (size_t i=0; i < Nvals; i++){
            for (size_t j=0; j < grid_points; j++){
                field_[i + j*Nvals] = values[i*grid_points + j];
            }
        }
    }
}

template<typename T, size_t NDIM>
RegularGridInterpolator<T, NDIM>::RegularGridInterpolator(const T* values, size_t Nvals, std::vector<MutView1D<T>> grid, bool values_axis_first) : x_(grid.size()){
    size_t nd = grid.size();
    assert((NDIM == 0 || nd == NDIM )&& "Number of grid arrays must match number of dimensions");
    Array1D<size_t, FDIM, Alloc> v_shape(nd+1);
    for (size_t i=0; i<nd; i++){
        v_shape[i] = grid[i].size();
        x_[i] = Array1D<T>(grid[i].data(), grid[i].shape(), 1, true);
    }
    v_shape[nd] = Nvals;
    field_.resize(v_shape.data(), v_shape.size());
    if (!values_axis_first){
        copy_array(field_.data(), values, field_.size());
    }else{
        size_t grid_points = prod(v_shape.data(), nd);
        assert(grid_points * Nvals == field_.size() && "Values array size not conserved, internal bug"); // sanity check
        for (size_t i=0; i < Nvals; i++){
            for (size_t j=0; j < grid_points; j++){
                field_[i + j*Nvals] = values[i*grid_points + j];
            }
        }
    }
}


template<typename T, size_t NDIM>
constexpr size_t RegularGridInterpolator<T, NDIM>::ndim() const{
    return x_.size();
}

template<typename T, size_t NDIM>
T RegularGridInterpolator<T, NDIM>::length(size_t axis) const{
    const Array1D<T>& x_grid = x_[axis];
    return x_grid[x_grid.size()-1] - x_grid[0];
}

template<typename T, size_t NDIM>
const Array1D<Array1D<T>, NDIM, RegularGridInterpolator<T, NDIM>::Alloc>& RegularGridInterpolator<T, NDIM>::x_all() const{
    return x_;
}

template<typename T, size_t NDIM>
const Array1D<T>& RegularGridInterpolator<T, NDIM>::x_data(size_t axis) const{
    assert(axis < this->ndim() && "Axis out of bounds");
    return x_[axis];
}

template<typename T, size_t NDIM>
bool RegularGridInterpolator<T, NDIM>::coords_in_bounds(const T* coords) const{
    assert(coords != nullptr && "Coords pointer cannot be null");
    if constexpr (NDIM > 0){
        return EXPAND(size_t, NDIM, I,
            return ((coords[I] >= x_[I][0] && coords[I] <= x_[I][x_[I].size()-1]) && ...);
        );
    }else{
        for (size_t axis=0; axis<this->ndim(); axis++){
            if (coords[axis] < x_[axis][0] || coords[axis] > x_[axis][x_[axis].size()-1]){
                return false;
            }
        }
        return true;
    }
}

template<typename T, size_t NDIM>
bool RegularGridInterpolator<T, NDIM>::value_in_axis(const T& x, size_t axis) const{
    assert(axis < this->ndim() && "Axis out of bounds");
    const Array1D<T>& X = x_[axis];
    return (x >= X[0] && x <= X[X.size()-1]);
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
const Array1D<T>& RegularGridInterpolator<T, NDIM>::x(size_t axis) const{
    assert(axis < this->ndim() && "Axis out of bounds");
    return x_[axis];
}


template<typename T, size_t NDIM>
void RegularGridInterpolator<T, NDIM>::interp_norm(T* out, const T* q) const{
    this->interp(out, q);
    size_t nfields = field_.shape(ndim());
    T norm = 0;
    for (size_t i=0; i<nfields; i++){
        norm += out[i]*out[i];
    }
    norm = sqrt(norm);
    for (size_t i=0; i<nfields; i++){
        out[i] /= norm;
    }
}

template<typename T, size_t NDIM>
void RegularGridInterpolator<T, NDIM>::interp(T* out, const T* q) const{
    size_t nd = this->ndim();
    Array2D<T, NDIM, 2, Alloc> cube(nd, 2); // cube(axis, neighbor)

    Array1D<T, NDIM, Alloc> coefs(nd); // interpolation coefficients for each axis
    size_t nfields = field_.shape(nd);

    for (size_t axis=0; axis<nd; axis++){
        size_t left_nbr = this->get_left_nbr(q[axis], axis);
        cube(axis, 0) = this->x_[axis][left_nbr];
        cube(axis, 1) = this->x_[axis][left_nbr+1];
        coefs[axis] = (q[axis] - cube(axis, 0))/(cube(axis, 1) - cube(axis, 0));
    }


    size_t n_corners = 1 << nd; // 2^ndim
    // initialize output to 0
    for (size_t field=0; field<nfields; field++){
        out[field] = 0;
    }

    // perform multilinear interpolation
    for (size_t corner = 0; corner < n_corners; corner++){
        T weight = 1;
        size_t offset = 0;
        for (size_t axis=0; axis<nd; axis++){
            size_t bit = (corner >> axis) & 1;
            weight *= (bit == 1) ? coefs[axis] : (1 - coefs[axis]);
            offset = offset * x_[axis].size() + (bit == 1 ? this->get_left_nbr(q[axis], axis) + 1 : this->get_left_nbr(q[axis], axis));
        }
        for (size_t field=0; field<nfields; field++){
            out[field] += weight * field_[offset*nfields + field];
        }
    }
}


} // namespace ode

#endif // GRID_INTERP_IMPL_HPP