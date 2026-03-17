#ifndef DISPATCHER_HPP
#define DISPATCHER_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <map>
#ifdef MPREAL
#include <mpreal.h>
#endif

namespace ode{

// All tested working scalar types should be included here
enum class ScalarType : std::uint8_t {
    Float,
    Double,
    LongDouble,
#ifdef MPREAL
    MPReal
#endif
};

static const std::map<ScalarType, const char*> SCALAR_TYPE_MAP = {
    {ScalarType::Float, "float"},
    {ScalarType::Double, "double"},
    {ScalarType::LongDouble, "long double"},
#ifdef MPREAL
    {ScalarType::MPReal, "mpreal"}
#endif
};

static const std::map<std::string, ScalarType> DTYPE_MAP = {
    {"float", ScalarType::Float},
    {"double", ScalarType::Double},
    {"long double", ScalarType::LongDouble},
#ifdef MPREAL
    {"mpreal", ScalarType::MPReal}
#endif
};

inline ScalarType getScalarType(const std::string& dtype){
    auto it = DTYPE_MAP.find(dtype);
    if (it != DTYPE_MAP.end()){
        return it->second;
    }
    else{
        throw std::runtime_error("Unsupported scalar type: " + dtype);
    }
}

inline const char* getScalarType(ScalarType st){
    auto it = SCALAR_TYPE_MAP.find(st);
    if (it != SCALAR_TYPE_MAP.end()){
        return it->second;
    }
    else{
        throw std::runtime_error("Invalid ScalarType enum value");
    }
}


template<typename T>
inline std::string get_scalar_type(){
    if constexpr (std::is_same_v<T, float>){
        return "float";
    }
    else if constexpr (std::is_same_v<T, double>){
        return "double";
    }
    else if constexpr (std::is_same_v<T, long double>){
        return "long double";
    }
#ifdef MPREAL
    else if constexpr (std::is_same_v<T, mpfr::mpreal>){
        return "mpreal";
    }
#endif
    else{
        static_assert(false, "Unsupported scalar type T");
    }
}

#define DISPATCH(RETURN_TYPE, ...)                                              \
call_dispatch(this->scalar_type, [&]<typename T>() -> RETURN_TYPE {__VA_ARGS__ });

// All implemented solvers must be included here
enum class Integrator : uint8_t{
    Euler,
    RK4,
    RK23,
    RK45,
    DOP853,
    BDF
};


inline const char* integrator_name(Integrator method){
    switch (method){
        case Integrator::Euler: return "Euler";
        case Integrator::RK4: return "RK4";
        case Integrator::RK23: return "RK23";
        case Integrator::RK45: return "RK45";
        case Integrator::DOP853: return "DOP853";
        case Integrator::BDF: return "BDF";
        default: throw std::runtime_error("Unknown integrator enum value");
    }
}


inline Integrator getIntegrator(const char* name){
    if (strcmp(name, "Euler") == 0){
        return Integrator::Euler;
    }else if (strcmp(name, "RK4") == 0){
        return Integrator::RK4;
    }else if (strcmp(name, "RK23") == 0){
        return Integrator::RK23;
    }else if (strcmp(name, "RK45") == 0){
        return Integrator::RK45;
    }else if (strcmp(name, "DOP853") == 0){
        return Integrator::DOP853;
    }else if (strcmp(name, "BDF") == 0){
        return Integrator::BDF;
    }else{
        throw std::runtime_error("Unknown integrator name");
    }
}

inline Integrator getIntegrator(const std::string& name){
    return getIntegrator(name.c_str());
}

} // namespace ode

#endif // DISPATCHER_HPP