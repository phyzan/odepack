#ifndef ND_INTERPOLATOR_IMPL_HPP
#define ND_INTERPOLATOR_IMPL_HPP


#include "NdInterpolator.hpp"

namespace ode {


template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
bool NdInterpolator<Derived, T, NDIM, AS_VIRTUAL>::norm_interp(T* out, const T* coords) const{
    if (!this->interp(out, coords)){
        return false;
    }
    int nfields = this->nvals_per_point();
    T norm = 0;
    for (int i=0; i<nfields; i++){
        norm += out[i]*out[i];
    }
    norm = sqrt(norm);
    for (int i=0; i<nfields; i++){
        out[i] /= norm;
    }
    return true;
}



template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
template<typename ArrayType>
NdInterpolator<Derived, T, NDIM, AS_VIRTUAL>::NdInterpolator(const ArrayType& values, bool coord_axis_first) : field_(values.data(), values.shape(), values.ndim()){
    const T* values_ptr = values.data();
    if (!coord_axis_first){
        int dims = int(values.ndim());
        const auto* shape = values.shape();
        std::vector<int> new_shape(dims);
        new_shape[0] = shape[dims-1];
        for (int i=1; i<dims; i++){
            new_shape[i] = shape[i-1];
        }
        field_.reshape(new_shape.data(), dims);
        int nvals = this->nvals_per_point();
        int np = this->n_points();
        for (int i=0; i < nvals; i++){
            for (int j=0; j < np; j++){
                field_[j*nvals + i] = values_ptr[i*np + j];
            }
        }
    } else {
        copy_array(field_.data(), values_ptr, field_.size());
    }

    if (field_.ndim() == 1){
        // add one more axis with size = 1
        std::vector<size_t> new_shape(field_.ndim() + 1);
        for (size_t i = 0; i < field_.ndim(); i++){
            new_shape[i] = field_.shape(i);
        }
        new_shape.back() = 1;
        field_.reshape(new_shape.data(), new_shape.size());
    }
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
bool NdInterpolator<Derived, T, NDIM, AS_VIRTUAL>::interp(T* out, const T* coords) const{
    return THIS->interp(out, coords);
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
int NdInterpolator<Derived, T, NDIM, AS_VIRTUAL>::ndim() const {
    return THIS->ndim();
}

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
bool NdInterpolator<Derived, T, NDIM, AS_VIRTUAL>::contains(const T* coords) const{
    return THIS->contains(coords);
}


} // namespace ode


#endif // ND_INTERPOLATOR_IMPL_HPP