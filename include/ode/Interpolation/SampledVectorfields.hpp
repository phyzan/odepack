#ifndef SAMPLED_VECTORFIELDS_HPP
#define SAMPLED_VECTORFIELDS_HPP

#include "GridInterp.hpp"
#include "../OdeInt.hpp"


namespace ode {


template<typename T, size_t NDIM>
class SampledVectorField : public RegularGridInterpolator<T, NDIM>{

    using Base = RegularGridInterpolator<T, NDIM>;

public:

    template<typename... Args>
    SampledVectorField(const Args&... args);

    INLINE auto ode_func_norm() const {
        return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
            for (size_t i = 0; i < NDIM; i++) {
                if (!this->value_in_axis(q[i], i)) {
                    for (size_t j = 0; j < NDIM; j++) {
                        out[j] = 0;
                    }
                    return;
                }
            }
            this->get_norm(out, q);
        };
    }

    INLINE auto ode_func() const {
        return [this](T* out, const T& t, const T* q, const T* args, const void* ptr){
            for (size_t i = 0; i < NDIM; i++) {
                if (!this->value_in_axis(q[i], i)) {
                    for (size_t j = 0; j < NDIM; j++){
                        out[j] = 0;
                    }
                    return;
                }
            }
            this->get(out, q);
        };
    }

    OdeResult<T>                        streamline(const T* x0, T length, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, const StepSequence<T>& t_eval, const std::string& method) const;

    std::vector<Array2D<T, NDIM, 0>>    streamplot_data(const T& max_length, const T& ds, size_t density) const;

private:

    template<size_t... I>
    std::vector<Array2D<T, NDIM, 0>>    streamplot_data_core(const T& max_length, const T& ds, size_t density, std::index_sequence<I...>) const;

};



} // namespace ode


#endif // SAMPLED_VECTORFIELDS_HPP