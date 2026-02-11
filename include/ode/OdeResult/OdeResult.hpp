#ifndef ODE_RESULT_HPP
#define ODE_RESULT_HPP

#include "../Tools.hpp"
#include "../Interpolation/StateInterp.hpp"

namespace ode{


//===========================================================================================
//                                      OdeResult<T, N>
//===========================================================================================

//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<typename T, size_t N=0>
class OdeResult{

public:

    OdeResult(const std::vector<T>& t, const Array2D<T, 0, N>& q, EventMap event_map, bool diverges, bool success, double runtime, std::string message);

    OdeResult() = default;

    DEFAULT_RULE_OF_FOUR(OdeResult);

    virtual ~OdeResult() = default;

    const std::vector<T>& t() const;

    const Array2D<T, 0, N>& q() const;

    template<std::integral INT1, std::integral INT2>
    const T& q(INT1 i, INT2 j) const;

    const EventMap& event_map() const;

    bool diverges() const;

    bool success() const;

    double runtime() const;

    const std::string& message() const;

    void examine() const;

    std::string event_log() const;

    std::vector<T> t_filtered(const std::string& event) const;

    Array2D<T, 0, N> q_filtered(const std::string& event) const;

    virtual OdeResult<T, N>* clone() const;
    
private:

    std::vector<T> _t;
    Array2D<T, 0, N> _q;
    EventMap _event_map;
    bool _diverges = false;
    bool _success = false;
    double _runtime = 0;
    std::string _message = "No integration performed";
};


// ============================================================================================
//                                      OdeSolution<T, N>
// ============================================================================================


template<typename T, size_t N=0>
class OdeSolution : public OdeResult<T, N>{

public:

    OdeSolution(const std::vector<T>& t, const Array2D<T, 0, N>& q, const EventMap& event_map, bool diverges, bool success, double runtime, std::string message, const Interpolator<T, N>& interpolator);

    OdeSolution(const OdeSolution& other);

    OdeSolution(OdeSolution&& other) noexcept;

    OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator);

    OdeSolution& operator=(const OdeSolution& other);

    OdeSolution& operator=(OdeSolution&& other) noexcept;

    ~OdeSolution();

    Array1D<T, N> operator()(const T& t) const;

    OdeSolution<T, N>* clone() const override;

private:

    Interpolator<T, N>* _interpolator;

};

}

#endif // ODE_RESULT_HPP