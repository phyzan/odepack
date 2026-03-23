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

    size_t size() const{
        return t.size();
    }

    View1D<T> t_view() const{
        return View1D<T>(t.data(), t.size());
    }

    View2D<T, 0, 0> q_view() const{
        return View2D<T, 0, 0>(q.data(), t.size(), nsys);
    }

    const T& get_q(size_t i, size_t j) const{
        assert(i < t.size() && j < nsys && "Index out of range");
        return q[i * nsys + j];
    }

    void clear_points(){
        t = std::vector<T>();
        q = std::vector<T>();
    }

    void add_point(const T& t_val, const T* q_val){
        t.push_back(t_val);
        try {
            q.insert(q.end(), q_val, q_val + nsys);
        } catch (...) {
            t.pop_back();  // rollback
            throw;
        }
    }
};


template<typename T>
class EventData{

public:

    EventData(size_t nsys) : nsys_(nsys) {}

    EventData() = default;

    EventData(const EventData& other, const std::vector<size_t>& start_indices) : nsys_(other.nsys_), event_names_(other.event_names_) {
        assert(start_indices.size() == other.size() && "Start indices size must match the number of events");
        for (size_t i = 0; i < other.size(); ++i) {
            size_t start_idx = start_indices[i];
            assert(start_idx <= other.data(i).t.size() && "Start index must be within the bounds of the provided data");
            OrbitData<T> data;
            const OrbitData<T>& other_data = other.data(i);
            data.t = std::vector<T>(other_data.t.begin() + start_idx, other_data.t.end());
            data.q = std::vector<T>(other_data.q.begin() + start_idx * nsys_, other_data.q.end());
            data.nsys = other_data.nsys;
            event_data_.push_back(std::move(data));
        }
    }
    DEFAULT_RULE_OF_FOUR(EventData)

    size_t size() const{
        return event_data_.size();
    }

    size_t nsys() const{
        return nsys_;
    }

    const OrbitData<T>& data(size_t i) const{
        assert(i < event_data_.size() && "Event index out of range");
        return event_data_[i];
    }

    const OrbitData<T>& data(const std::string& name) const{
        auto it = std::ranges::find(event_names_, name);
        if (it == event_names_.end()){
            throw std::runtime_error("Event name not found: " + name);
        }
        size_t index = std::distance(event_names_.begin(), it);
        return this->data(index);
    }

    const std::string& name(size_t i) const{
        assert(i < event_names_.size() && "Event index out of range");
        return event_names_[i];
    }

    void allocate_event(const std::string& name){
        // check that name is not empty and not already in the map
        if (name.empty()){
            throw std::runtime_error("Event name cannot be empty");
        }
        if (std::ranges::find(event_names_, name) != event_names_.end()){
            throw std::runtime_error("Event name already exists");
        }
        event_data_.emplace_back(OrbitData<T>{.nsys = nsys_});

        try {
            event_names_.push_back(name);
        } catch (...) {
            event_data_.pop_back();  // rollback
            throw;
        }
    }

    void add_event(size_t event_idx, const T& t, const T* q){
        if (event_idx >= event_names_.size()){
            throw std::runtime_error("Event index out of range");
        }
        OrbitData<T>& data = event_data_[event_idx];
        data.t.push_back(t);
        try {
            data.q.insert(data.q.end(), q, q + nsys_);
        } catch (...) {
            data.t.pop_back();  // rollback
            throw;
        }
    }


    void add_event(const std::string& name, const T& t, const T* q){
        auto it = std::ranges::find(event_names_, name);
        if (it == event_names_.end()){
            throw std::runtime_error("Event name not found: " + name);
        }
        size_t index = std::distance(event_names_.begin(), it);
        this->add_event(index, t, q);
    }

    void clear_points(){
        for (auto& data : event_data_){
            data = OrbitData<T>{.nsys = nsys_};
        }
    }

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

    PolyWrapper<Interpolator<T, N>> interpolator_;

};

}

#endif // ODE_RESULT_HPP