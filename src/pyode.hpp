#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include "ode.hpp"

namespace py = pybind11;

template<class Tt, class Ty>
Func<Tt, Ty> to_Func(py::object f, const _Shape& shape);

_Shape shape(const py::array& arr);

_Shape getShape(const size_t& dim1, const _Shape& shape){
    std::vector<size_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const _Shape& _shape);

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(py::object py_event, const _Shape& shape);

template<class Tt, class Ty>
is_event_f<Tt, Ty> to_event_check(py::object py_event_check, const _Shape& shape);

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map);

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A);

template<class Tt, class Ty>
std::vector<Tt> flatten(const std::vector<Ty>&);

template<class T>
py::tuple to_tuple(const std::vector<T>& vec);

template<class Tt, class Ty>
std::vector<Event<Tt, Ty>> to_Events(py::object events, const _Shape& shape);


template<class Tt, class Ty>
std::vector<StopEvent<Tt, Ty>> to_StopEvents(py::object events, const _Shape& shape);


#pragma GCC visibility push(hidden)


template<class Tt, class Ty>
struct PyEvent{

    PyEvent(py::str name, py::object when, py::object check_if, py::object mask): _name(name.cast<std::string>()), py_when(when), py_check_if(check_if), py_mask(mask){}

    std::string _name;
    py::object py_when;
    py::object py_check_if;
    py::object py_mask;

    Event<Tt, Ty> toEvent(const _Shape& shape){
        return Event<Tt, Ty>(_name, to_event<Tt, Ty>(py_when, shape), to_event_check<Tt, Ty>(py_check_if, shape), to_Func<Tt, Ty>(py_mask, shape));
    }
};



template<class Tt, class Ty>
struct PyStopEvent{

    PyStopEvent(py::str name, py::object when, py::object check_if): _name(name.cast<std::string>()), py_when(when), py_check_if(check_if){}

    std::string _name;
    py::object py_when;
    py::object py_check_if;

    StopEvent<Tt, Ty> toStopEvent(const _Shape& shape){
        return StopEvent<Tt, Ty>(_name, to_event<Tt, Ty>(py_when, shape), to_event_check<Tt, Ty>(py_check_if, shape));
    }
};


template<class Tt, class Ty>
class PySolverState : public SolverState<Tt, Ty>{

public:
    PySolverState(const Tt& t, const Ty& q, const Tt& habs, const std::string& event, const bool& diverges, const bool& is_stiff, const bool& is_running, const bool& is_dead, const size_t& N, const std::string& message, const _Shape& shape): SolverState<Tt, Ty>(t, q, habs, event, diverges, is_stiff, is_running, is_dead, N, message), shape(shape) {}

    const std::vector<size_t> shape;

};


template<class Tt, class Ty>
struct PyOdeResult{

    PyOdeResult(const OdeResult<Tt, Ty>& r, const _Shape& q0_shape): res(r), q0_shape(q0_shape){}

    const OdeResult<Tt, Ty> res;
    const _Shape& q0_shape;

};


template<class Tt, class Ty>
class PyODE : public ODE<Tt, Ty>{
public:

    PyODE(py::object f, const Tt t0, const py::array q0, const Tt stepsize, const Tt rtol, const Tt atol, const Tt min_step, const py::tuple args, const py::str method, const Tt event_tol, py::object events, py::object stop_events) : ODE<Tt, Ty>(to_Func<Tt, Ty>(f), shape(q0), t0, toCPP_Array<Tt, Ty>(q0), stepsize, rtol, atol, min_step, toCPP_Array<Tt, std::vector<Tt>>(args), method.cast<std::string>(), event_tol, to_Events(events), to_StopEvents(stop_events)), _shape(shape(q0)){}

    PyODE(const Func<Tt, Ty> f, const Tt t0, const Ty q0, const Tt stepsize, const Tt rtol, const Tt atol, const Tt min_step, const std::vector<Tt> args = {}, const std::string& method = "RK45", const Tt event_tol = 1e-10, const std::vector<Event<Tt, Ty>>& events = {}, const std::vector<StopEvent<Tt, Ty>>& stop_events = {}) : ODE<Tt, Ty>(f, t0, q0, stepsize, rtol, atol, min_step, args, method, event_tol, events, stop_events), _shape({q0.size()}){}

    PySolverState<Tt, Ty> py_state() const{
        SolverState<Tt, Ty> s = ode.state();
        return PySolverState<Tt, Ty>(s.t, s.q, s.habs, s.event, s.diverges, s.is_stiff, s.is_running, s.is_dead, s.N, s.message, _shape); 
    }

private:
    const std::vector<size_t> _shape;
};

#pragma GCC visibility pop


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/


template<class Tt, class Ty>
Func<Tt, Ty> to_Func(py::object f, const _Shape& shape){
    if (f.is_none()){
        return nullptr;
    }

    Func<Tt, Ty> g = [f, shape](const Tt& t, const Ty& y, const std::vector<Tt>& args) -> Ty {
        return toCPP_Array<Tt, Ty>(f(t, to_numpy<Tt>(y, shape), *to_tuple(args)));
    };
    return g;
}

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(py::object py_event, const _Shape& shape){
    if (py_event.is_none()){
        return nullptr;
    }
    event_f<Tt, Ty> g = [py_event, shape](const Tt& t, const Ty& f, const std::vector<Tt>& args) -> Tt {
        return py_event(t, to_numpy<Tt>(f, shape), *to_tuple(args)).template cast<Tt>();
    };
    return g;
}

template<class Tt, class Ty>
is_event_f<Tt, Ty> to_event_check(py::object py_event_check, const _Shape& shape){
    if (py_event_check.is_none()){
        return nullptr;
    }

    is_event_f<Tt, Ty> g = [py_event_check, shape](const Tt& t, const Ty& f, const std::vector<Tt>& args) -> bool {
        return py_event_check(t, to_numpy<Tt>(f, shape), *to_tuple(args)).equal(py::bool_(true));
    };
    return g;
}

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A){
    size_t n = A.size();
    Ty res(n);

    const Tt* data = static_cast<const Tt*>(py::float_(A.data()));

    for (size_t i=0; i<n; i++){
        res[i] = data[i];
    }
    return res;
}

_Shape shape(const py::array& arr) {
    const ssize_t* shape_ptr = arr.shape();  // Pointer to shape data
    size_t ndim = arr.ndim();  // Number of dimensions
    std::vector<size_t> res(shape_ptr, shape_ptr + ndim);
    return res;
}

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map){
    py::dict py_dict;
    for (const auto& [key, vec] : _map) {
        py::array_t<size_t> np_array(vec.size(), vec.data()); // Create NumPy array
        py_dict[key.c_str()] = np_array; // Assign to dictionary
    }
    return py_dict;
}

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const _Shape& _shape){
    if (_shape.size() == 0){
        py::array_t<Scalar> res(array.size(), array.data());
        return res;
    }
    else{
        py::array_t<Scalar> res(_shape, array.data());
        return res;
    }
}

template<class Tt, class Ty>
std::vector<Tt> flatten(const std::vector<Ty>& f){
    size_t nt = f.size();
    if (nt == 0){
        return {};
    }
    size_t nd = f[0].size();
    std::vector<Tt> res(nt*nd);

    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<nd; j++){
            res[i*nd + j] = f[i][j];
        }
    }
    return res;
}

template<class T>
py::tuple to_tuple(const std::vector<T>& arr) {
    py::tuple py_tuple(arr.size());  // Create a tuple of the same size as the vector
    for (size_t i = 0; i < arr.size(); ++i) {
        py_tuple[i] = py::float_(arr[i]);  // Convert each double element to py::float_
    }
    return py_tuple;
}

template<class Tt, class Ty>
std::vector<Event<Tt, Ty>> to_Events(py::object events, const _Shape& shape){   
    if (events.is_none()){
        return {};
    }
    std::vector<Event<Tt, Ty>> res;
    for (const py::handle& item : events){
        res.push_back(item.cast<PyEvent<Tt, Ty>&>().toEvent(shape));
    }
    return res;
}


template<class Tt, class Ty>
std::vector<StopEvent<Tt, Ty>> to_StopEvents(py::object events, const _Shape& shape){

    if (events.is_none()){
        return {};
    }
    
    std::vector<StopEvent<Tt, Ty>> res;
    for (const py::handle& item : events){
        res.push_back(item.cast<PyStopEvent<Tt, Ty>&>().toStopEvent(shape));
    }
    return res;
}



template<class Tt, class Ty>
void define_ode_module(py::module& m) {
    py::class_<PyEvent<Tt, Ty>>(m, "Event", py::module_local())
        .def(py::init<py::str, py::object, py::object, py::object>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("check_if")=py::none(),
            py::arg("mask")=py::none());

    py::class_<StopEvent<Tt, Ty>>(m, "StopEvent", py::module_local())
        .def(py::init<py::str, py::object, py::object>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("check_if")=py::none());

    py::class_<PyOdeResult<Tt, Ty>>(m, "OdeResult", py::module_local())
        .def_property_readonly("t", [](const PyOdeResult<Tt, Ty>& self){
            return to_numpy<Tt>(self.res.t);
        })
        .def_property_readonly("q", [](const PyOdeResult<Tt, Ty>& self){
            return to_numpy<Tt>(flatten<Tt, Ty>(self.res.q), getShape(self.res.t.size(), self.q0_shape));
        })
        .def_property_readonly("events", [](const PyOdeResult<Tt, Ty>& self){
            return to_PyDict(self.res.events);
        })
        .def_property_readonly("diverges", [](const PyOdeResult<Tt, Ty>& self){return self.res.diverges;})
        .def_property_readonly("is_stiff", [](const PyOdeResult<Tt, Ty>& self){return self.res.is_stiff;})
        .def_property_readonly("success", [](const PyOdeResult<Tt, Ty>& self){return self.res.success;})
        .def_property_readonly("runtime", [](const PyOdeResult<Tt, Ty>& self){return self.res.runtime;})
        .def_property_readonly("message", [](const PyOdeResult<Tt, Ty>& self){return self.res.message;})
        .def("examine", [](const PyOdeResult<Tt, Ty>& self){return self.res.examine();});


    py::class_<PySolverState<Tt, Ty>>(m, "SolverState", py::module_local())
        .def_property_readonly("t", [](const PySolverState<Tt, Ty>& self){return self.t;})
        .def_property_readonly("q", [](const PySolverState<Tt, Ty>& self){return to_numpy<Tt>(self.q, self.shape);})
        .def_property_readonly("event", [](const PySolverState<Tt, Ty>& self){return self.event;})
        .def_property_readonly("diverges", [](const PySolverState<Tt, Ty>& self){return self.diverges;})
        .def_property_readonly("is_stiff", [](const PySolverState<Tt, Ty>& self){return self.is_stiff;})
        .def_property_readonly("is_running", [](const PySolverState<Tt, Ty>& self){return self.is_running;})
        .def_property_readonly("is_dead", [](const PySolverState<Tt, Ty>& self){return self.is_dead;})
        .def_property_readonly("N", [](const PySolverState<Tt, Ty>& self){return self.N;})
        .def_property_readonly("message", [](const PySolverState<Tt, Ty>& self){return self.message;})
        .def("show", [](const PySolverState<Tt, Ty>& self){return self.show();});

        

    py::class_<PyODE<Tt, Ty>>(m, "LowLevelODE", py::module_local())
        .def(py::init<py::object, Tt, py::array, Tt, Tt, Tt, Tt, py::tuple, py::str, Tt, py::object, py::object>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("stepsize"),
            py::kw_only(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("event_tol")=1e-12,
            py::arg("events")=py::none(),
            py::arg("stop_events")=py::none())
        .def("integrate", [](const PyODE<Tt, Ty>& self, const Tt& interval, const int max_frames, const int& max_events, const bool& terminate, const bool& display){return PyOdeResult<Tt, Ty>(self.integrate(interval, max_frames, max_events, terminate, display), self._shape)},
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("max_events")=-1,
            py::arg("terminate")=true,
            py::arg("display")=false)
        .def("copy", [](const PyODE<Tt, Ty>& self){return PyODE<Tt, Ty>(self);})
        .def("advance", [](PyODE<Tt, Ty>& self){return self.advance();})
        .def("resume", [](PyODE<Tt, Ty>& self){return self.resume();})
        .def("free", [](PyODE<Tt, Ty>& self){return self.free();})
        .def("state", &PyODE<Tt, Ty>::py_state)
        .def_property_readonly("t", [](const PyODE<Tt, Ty>& self){return to_numpy<Tt, Ty>(self.t());})
        .def_property_readonly("q", [](const PyODE<Tt, Ty>& self){return to_numpy<Tt, Ty>(flatten<Tt, Ty>(self.q()), self._shape);})
        .def_property_readonly("event_map", [](const PyODE<Tt, Ty>& self){return to_PyDict(self.event_map());})
        .def_property_readonly("runtime", &PyODE<Tt, Ty>::runtime)
        .def_property_readonly("is_stiff", &PyODE<Tt, Ty>::is_stiff)
        .def_property_readonly("diverges", &PyODE<Tt, Ty>::diverges)
        .def_property_readonly("is_dead", &PyODE<Tt, Ty>::is_dead);


    m.def("integrate_all", [](py::object list, const Tt& interval, const int& max_frames, const int& max_events, const bool& terminate){
        std::vector<ODE<Tt, Ty>*> array;
        for (const py::handle& item : list){
            array.push_back(&(item.cast<ODE<Tt, Ty>&>()));
        }
        integrate_all(array, interval, max_frames, max_events, terminate);
    }, py::arg("ode_array"), py::arg("interval"), py::arg("max_frames")=-1, py::arg("max_events")=-1, py::arg("terminate")=true);
}


//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

#endif
