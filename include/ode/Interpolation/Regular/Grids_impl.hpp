#ifndef GRIDS_IMPL_HPP
#define GRIDS_IMPL_HPP

#include "Grids.hpp"

namespace ode{

template<typename T, int NDIM>
template<typename AxisData>
RegularGrid<T, NDIM>::RegularGrid(const AxisData& axes) : data_(axes.size()), shape_(axes.size()) {
    assert((NDIM == 0 || axes.size() == NDIM) && "Number of axes must match NDIM");
    int n_axes = axes.size();
    for (int i=0; i<n_axes; i++){
        const auto& axis = axes[i];
        assert(axis.size() >= 2 && "Each axis must have at least 2 grid points");
        data_[i] = Array1D<T>(axis.data(), axis.size());
        shape_[i] = axis.size();
    }
}

template<typename T, int NDIM>
int RegularGrid<T, NDIM>::ndim() const{
    if constexpr (NDIM > 0){
        return NDIM;
    }else {
        return int(data_.size());
    }
}

template<typename T, int NDIM>
T RegularGrid<T, NDIM>::length(int axis) const {
    assert((axis >= 0 && axis < ndim()) && "Axis index out of bounds");
    const auto& arr = data_[axis];
    return arr[arr.size()-1] - arr[0];
}

template<typename T, int NDIM>
const Array1D<T>& RegularGrid<T, NDIM>::x(int i) const {
    assert((i >= 0 && i < ndim()) && "Axis index out of bounds");
    return data_[i];
}

template<typename T, int NDIM>
bool RegularGrid<T, NDIM>::contains(const T& x, int axis) const {
    assert((axis >= 0 && axis < ndim()) && "Axis index out of bounds");
    const auto& arr = data_[axis];
    return x >= arr[0] && x <= arr[arr.size()-1];
}

template<typename T, int NDIM>
bool RegularGrid<T, NDIM>::contains(const T* coords) const {
    for (int i=0; i<ndim(); i++){
        if (!contains(coords[i], i)){
            return false;
        }
    }
    return true;
}

template<typename T, int NDIM>
int RegularGrid<T, NDIM>::find_left_idx(const T& x, int axis) const {
    assert((axis >= 0 && axis < ndim()) && "Axis index out of bounds");
    const T* arr = data_[axis].data();
    int size = int(data_[axis].size());
    if (!contains(x, axis)){
        return -1; // Out of bounds
    }
    // Binary search for the right interval
    int left = 0;
    int right = size - 1;
    while (left < right){
        int mid = left + (right - left) / 2;
        if (arr[mid] <= x){
            left = mid + 1;
        }else {
            right = mid;
        }
    }
    return left - 1;
}

 
} // namespace ode

#endif // GRIDS_IMPL_HPP