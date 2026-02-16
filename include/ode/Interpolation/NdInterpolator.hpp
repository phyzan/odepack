#ifndef ND_INTERPOLATOR_HPP
#define ND_INTERPOLATOR_HPP


#include "../Tools.hpp"

namespace ode {

struct VirtualNdInterpolator {

    virtual int ndim() const = 0;

    virtual int nvals_per_point() const = 0;

    virtual bool interp(double* out, const double* coords) const = 0;

    virtual const size_t* output_shape() const = 0;

    virtual size_t output_dims() const = 0;

}; // class VirtualNdInterpolator


struct EmptyNdInterpolator {};

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL = false>
class NdInterpolator : public std::conditional_t<AS_VIRTUAL, VirtualNdInterpolator, EmptyNdInterpolator> {


public:

    static constexpr Allocation ALLOC = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);

    inline int nvals_per_point() const { return int(field_.size() / n_points()); }

    inline int n_points() const { return field_.shape(0); }

    inline const Array<T>& values() const { return field_; }

    inline const size_t* output_shape() const { return field_.shape()+1; }

    inline size_t output_dims() const {return field_.ndim()-1;}

    bool norm_interp(T* out, const T* coords) const;

    // ============== Static Override ==================
    bool            interp(T* out, const T* coords) const;

    int             ndim() const;

    bool            contains(const T* coords) const;
    // =================================================

protected:

    template<typename ArrayType>
    NdInterpolator(const ArrayType& values, bool coord_axis_first);

private:
    
    Array<T> field_; // shape = (n_points, ...)
};

} // namespace ode


#endif // ND_INTERPOLATOR_HPP