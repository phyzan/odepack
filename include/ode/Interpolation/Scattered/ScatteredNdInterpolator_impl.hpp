#ifndef SCATTERED_ND_INTERPOLATOR_IMPL_HPP
#define SCATTERED_ND_INTERPOLATOR_IMPL_HPP


#include "ScatteredNdInterpolator.hpp"

namespace ode {


// ============================================================================================
//                              ScatteredNdInterpolator
// ============================================================================================


template<int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer>
ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::ScatteredNdInterpolator(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first) : Base(values, coord_axis_first), tri_(std::make_shared<DelaunayTri<NDIM>>(points, values.shape(coord_axis_first ? 0 : values.ndim()-1), ndim)) {}

template<int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer>
ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::ScatteredNdInterpolator(TriPtr<NDIM> tri, const ValuesContainer& values, bool coord_axis_first) : Base(values, coord_axis_first), tri_(tri) {
    assert(tri->npoints() == values.shape(coord_axis_first ? 0 : values.ndim()-1) && "Mismatch between values and Delaunay points");
}

template<int NDIM, bool AS_VIRTUAL>
bool ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::interp(double* out, const double* coords) const{
    return tri_->interpolate(out, coords, this->values().data(), this->nvals_per_point());
}

template<int NDIM, bool AS_VIRTUAL>
int ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::ndim() const {
    return tri_->ndim();
}

template<int NDIM, bool AS_VIRTUAL>
bool ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::contains(const double* coords) const {
    return tri_->contains(coords);
}

template<int NDIM, bool AS_VIRTUAL>
TriPtr<NDIM> ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::tri() const { return tri_; }

template<int NDIM, bool AS_VIRTUAL>
const Array2D<double, 0, NDIM>& ScatteredNdInterpolator<NDIM, AS_VIRTUAL>::points() const {
    return tri_->get_points();
}


// ============================================================================================
//                              ScatteredVectorField
// ============================================================================================

template<int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer>
ScatteredVectorField<NDIM, AS_VIRTUAL>::ScatteredVectorField(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first) : InterpBase(points, values, ndim, coord_axis_first), VFBase() {
    assert((NDIM == 0 || ndim == NDIM) && "ndim must match template NDIM");
}

template<int NDIM, bool AS_VIRTUAL>
template<typename ValuesContainer>
ScatteredVectorField<NDIM, AS_VIRTUAL>::ScatteredVectorField(const TriPtr<NDIM>& tri, const ValuesContainer& values, bool coord_axis_first) : InterpBase(tri, values, coord_axis_first), VFBase() {
    assert((NDIM == 0 || tri->ndim() == NDIM) && "Delaunay ndim must match template NDIM");
}

template<int NDIM, bool AS_VIRTUAL>
bool ScatteredVectorField<NDIM, AS_VIRTUAL>::interp(double* out, const double* coords) const{
    return InterpBase::interp(out, coords);
}

template<int NDIM, bool AS_VIRTUAL>
int ScatteredVectorField<NDIM, AS_VIRTUAL>::ndim() const {
    return InterpBase::ndim();
}

template<int NDIM, bool AS_VIRTUAL>
bool ScatteredVectorField<NDIM, AS_VIRTUAL>::contains(const double* coords) const {
    return InterpBase::contains(coords);
}

} // namespace ode

#endif // SCATTERED_ND_INTERPOLATOR_IMPL_HPP