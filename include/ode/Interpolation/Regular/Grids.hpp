#ifndef GRIDS_HPP
#define GRIDS_HPP

#include "../../../ndspan.hpp"

namespace ode{

using namespace ndspan;

template<typename T, int NDIM>
class RegularGrid{

    static_assert(NDIM >= 0, "NDIM must be non-negative");

    static constexpr Allocation ALLOC = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);

public:

    /*
    axes.size()     : ndim. If NDIM is known at compile time, this is checked to match NDIM in debug mode, otherwise it can be any positive integer.
    axis[i].size()  : number of grid points along axis i
    axis[i].data()  : pointer to the grid points along axis i
    */
    template<typename AxisData>
    RegularGrid(const AxisData& axes);

    int                 ndim() const;

    T                   length(int axis) const;

    const Array1D<T>&   x(int i) const;

    bool                contains(const T& x, int axis) const;

    bool                contains(const T* coords) const;

    int                 find_left_idx(const T& x, int axis) const;

    inline const auto&  data() const {return data_;}

    inline const int*   shape() const {return shape_.data();}

private:

    std::vector<Array1D<T>> data_;
    std::vector<int> shape_;
};

} // namespace ode

#endif // GRIDS_HPP