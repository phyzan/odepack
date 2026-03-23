#ifndef ODE_RESULT_HPP
#define ODE_RESULT_HPP

#include <algorithm>

#include "../Tools.hpp"
#include "../Interpolation/Univariate/StateInterp.hpp"

namespace ode{


//===========================================================================================
//                                      OdeResult<T, N>
//===========================================================================================

//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<typename T>
struct OrbitData{
    std::vector<T> t;
    std::vector<T> q; //flattened 2D array
    size_t nsys;

    size_t size() const;

    View1D<T> t_view() const;

    View2D<T, 0, 0> q_view() const;

    const T& get_q(size_t i, size_t j) const;

    void clear_points();

    void add_point(const T& t_val, const T* q_val);
};


template<typename T>
class EventData{

public:

    EventData(size_t nsys) : nsys_(nsys) {}

    EventData() = default;

    EventData(const EventData& other, const std::vector<size_t>& start_indices);

    DEFAULT_RULE_OF_FOUR(EventData)

    size_t size() const;

    size_t nsys() const;

    const OrbitData<T>& data(size_t i) const;

    const OrbitData<T>& data(const std::string& name) const;

    const std::string& name(size_t i) const;

    void allocate_event(const std::string& name);

    void add_event(size_t event_idx, const T& t, const T* q);

    void add_event(const std::string& name, const T& t, const T* q);

    void clear_points();

private:

    std::vector<OrbitData<T>> event_data_;
    std::vector<std::string> event_names_;
    size_t nsys_ = 0;
};


template<typename T, size_t N=0>
class OdeResult{

public:

    OdeResult(const OrbitData<T>& orbit_data, EventData<T> event_data, size_t orb_idx_start, bool diverges, bool success, double runtime, std::string message);

    OdeResult() = default;

    DEFAULT_RULE_OF_FOUR(OdeResult);

    virtual ~OdeResult() = default;

    View1D<T> t() const;

    View2D<T, 0, N> q() const;

    const T& q(size_t i, size_t j) const;

    const EventData<T>& event_data() const;

    bool diverges() const;

    bool success() const;

    double runtime() const;

    const std::string& message() const;

    void examine() const;

    std::string event_log() const;

    virtual OdeResult<T, N>* clone() const;

private:

    OrbitData<T> orbit_data_;
    EventData<T> event_data_;
    bool diverges_ = false;
    bool success_ = false;
    double runtime_ = 0;
    std::string message_ = "No integration performed";
};


// ============================================================================================
//                                      OdeSolution<T, N>
// ============================================================================================


template<typename T, size_t N=0>
class OdeSolution : public OdeResult<T, N>{

public:

    OdeSolution(OrbitData<T> orbit_data, EventData<T> event_data, size_t orb_idx_start, bool diverges, bool success, double runtime, std::string message, const Interpolator<T, N>& interpolator);

    OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator);

    Array1D<T, N> operator()(const T& t) const;

    OdeSolution<T, N>* clone() const override;

private:

    InterpObj<T, N> interpolator_;

};

}

#endif // ODE_RESULT_HPP
