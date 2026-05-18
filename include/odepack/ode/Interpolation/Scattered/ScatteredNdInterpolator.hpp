#ifndef SCATTERED_ND_INTERPOLATOR_HPP
#define SCATTERED_ND_INTERPOLATOR_HPP


#include "Delaunay.hpp"
#include "../NdInterpolator.hpp"
#include "../VectorFields.hpp"

namespace ode {

template<int NDIM, bool AS_VIRTUAL = false>
class ScatteredNdInterpolator : public NdInterpolator<ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>{

    using Base = NdInterpolator<ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>;

public:

    // points: shape (n_points, ndim)
    // values: shape (n_points, ...) if coord_axis_first, or (..., n_points) otherwise
    template<typename ValuesContainer>
    ScatteredNdInterpolator(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first);

    template<typename ValuesContainer>
    ScatteredNdInterpolator(TriPtr<NDIM> tri, const ValuesContainer& values, bool coord_axis_first);

    //========= Static Overrides ==============
    int             ndim() const;
    bool            interp(double* out, const double* coords) const;
    bool            contains(const double* coords) const;
    //=========================================

    TriPtr<NDIM> tri() const;

    // shape (n_points, ndim)
    const Array2D<double, 0, NDIM>& points() const;

private:

    TriPtr<NDIM> tri_;

}; // ScatteredNdInterpolator



template<int NDIM, bool AS_VIRTUAL>
class ScatteredVectorField : public ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, public VectorField<ScatteredVectorField<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>{

    using InterpBase = ScatteredNdInterpolator<NDIM, AS_VIRTUAL>;
    using VFBase = VectorField<ScatteredVectorField<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>;
public:

    // points: (n_points, ndim)
    // values: (n_points, ndim) if coord_axis_first, else (ndim, npoints)
    template<typename ValuesContainer>
    ScatteredVectorField(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first);

    template<typename ValuesContainer>
    ScatteredVectorField(const TriPtr<NDIM>& tri, const ValuesContainer& values, bool coord_axis_first);

    // ============ Explicit overrides for VectorField ==============
    bool interp(double* out, const double* coords) const;
    int ndim() const;
    bool contains(const double* coords) const;
    // ==============================================================

};

} // namespace ode

#endif // SCATTERED_ND_INTERPOLATOR_HPP