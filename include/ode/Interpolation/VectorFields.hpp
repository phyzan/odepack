#ifndef VECTOR_FIELDS_HPP
#define VECTOR_FIELDS_HPP


#include "../Tools.hpp"
#include "../OdeInt.hpp"

namespace ode {

struct VirtualVectorField {

    virtual ~VirtualVectorField() = default;

    virtual int ndim() const = 0;

    virtual bool contains(const double* coords) const = 0;

    virtual OdeResult<double> streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, const StepSequence<double>& t_eval, Integrator method, bool normalized) const = 0;

    virtual ODE<double, 0>* get_streamline_ode(const double* x0, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const = 0;

}; // class VirtualVectorField

struct EmptyVectorField {};

template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL>
class VectorField : public std::conditional_t<AS_VIRTUAL, VirtualVectorField, EmptyVectorField> {


public:

    // ============== Static Overrides ==================
    bool interp(T* out, const T* coords) const;
    int ndim() const;
    bool contains(const T* coords) const;
    // =================================================


    static void ode_func_norm(T* out, const T& t, const T* q, const T* args, const void* ptr);

    static void ode_func(T* out, const T& t, const T* q, const T* args, const void* ptr);

    OdeResult<T> streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, Integrator method, bool normalized) const;

    ODE<T, NDIM>*   get_streamline_ode(const T* x0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, Integrator method, bool normalized) const;

protected:

    VectorField() = default;

    DEFAULT_RULE_OF_FOUR(VectorField)

}; // class VectorField

} // namespace ode

#endif // VECTOR_FIELDS_HPP