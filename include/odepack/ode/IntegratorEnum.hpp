#ifndef INTEGRATORENUM_HPP
#define INTEGRATORENUM_HPP

#include <cstdint>
#include <stdexcept>

namespace ode{

enum class Integrator : uint8_t{
    Custom,
    Euler,
    RK4,
    RK23,
    RK45,
    DOP853,
    BDF
};


template<typename ReturnType, typename Callable, typename... Args>
inline ReturnType choose_integrator_case(Integrator method, Callable&& callable, Args&&... args){
    switch (method){
        case Integrator::Euler:
            return callable.template operator()<Integrator::Euler>(std::forward<Args>(args)...);
        case Integrator::RK4:
            return callable.template operator()<Integrator::RK4>(std::forward<Args>(args)...);
        case Integrator::RK23:
            return callable.template operator()<Integrator::RK23>(std::forward<Args>(args)...);
        case Integrator::RK45:
            return callable.template operator()<Integrator::RK45>(std::forward<Args>(args)...);
        case Integrator::DOP853:
            return callable.template operator()<Integrator::DOP853>(std::forward<Args>(args)...);
        case Integrator::BDF:
            return callable.template operator()<Integrator::BDF>(std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unknown integrator enum value");
    }
}

} // namespace ode


#endif // INTEGRATORENUM_HPP