#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include "ode.hpp"

namespace py = pybind11;

//assumes empty args given as std::vector, use only for ODE instanciated from python directly
template<class Tt, class Ty>
Func<Tt, Ty> to_Func(py::object f, const _Shape& shape, py::tuple args=py::tuple());

_Shape shape(const py::array& arr);

_Shape getShape(const size_t& dim1, const _Shape& shape){
    std::vector<size_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const _Shape& q0_shape={});

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(py::object py_event, const _Shape& shape, py::tuple args);

template<class Tt, class Ty>
is_event_f<Tt, Ty> to_event_check(py::object py_event_check, const _Shape& shape, py::tuple py_args);

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map);

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A);

template<class Tt, class Ty>
Ty fast_convert(const py::array_t<Tt>& A);

template<class Tt, class Ty>
std::vector<Tt> flatten(const std::vector<Ty>&);

template<class T>
py::tuple to_tuple(const std::vector<T>& vec);

template<class Tt, class Ty>
std::vector<AnyEvent<Tt, Ty>*> to_Events(py::object events, const _Shape& shape, py::tuple args);



#pragma GCC visibility push(hidden)


template<class Tt, class Ty>
class PyAnyEvent{

public:
    PyAnyEvent(py::str name, py::object when, py::object check_if, py::object mask, py::bool_ hide_mask): _name(name.cast<std::string>()), py_when(when), py_check_if(check_if), py_mask(mask), hide_mask(hide_mask){}

    virtual ~PyAnyEvent(){
        delete this->_event_ptr;
        this->_event_ptr = nullptr;
    }

    virtual AnyEvent<Tt, Ty>* toEvent(const _Shape& shape, py::tuple args) = 0;

    py::str name()const{
        return py::str(_name);
    }

protected:
    std::string _name;

public:
    py::object py_when;
    py::object py_check_if;
    py::object py_mask;
    bool hide_mask;
    
    AnyEvent<Tt, Ty>* _event_ptr = nullptr;
};


template<class Tt, class Ty>
class PyEvent : public PyAnyEvent<Tt, Ty>{

public:
    PyEvent(py::str name, py::object when, py::object check_if, py::object mask, py::bool_ hide_mask, Tt event_tol):PyAnyEvent<Tt, Ty>(name, when, check_if, mask, hide_mask){}

    AnyEvent<Tt, Ty>* toEvent(const _Shape& shape, py::tuple args=py::tuple()) override{
        delete this->_event_ptr;
        this->_event_ptr = new Event<Tt, Ty>(this->_name, to_event<Tt, Ty>(this->py_when, shape, args), to_event_check<Tt, Ty>(this->py_check_if, shape, args), to_Func<Tt, Ty>(this->py_mask, shape, args), this->hide_mask, this->_event_tol);
        return this->_event_ptr;
    }

private:
    Tt _event_tol;

};


template<class Tt, class Ty>
class PyPerEvent : public PyAnyEvent<Tt, Ty>{

public:
    PyPerEvent(py::str name, const Tt& period, const Tt& start, py::object mask, py::bool_ hide_mask):PyAnyEvent<Tt, Ty>(name, py::none(), py::none(), mask, hide_mask), _period(period), _start(start){}

    AnyEvent<Tt, Ty>* toEvent(const _Shape& shape, py::tuple args) override{
        delete this->_event_ptr;
        this->_event_ptr = new PeriodicEvent<Tt, Ty>(this->_name, this->_period, this->_start, to_Func<Tt, Ty>(this->py_mask, shape, args), this->hide_mask);
        return this->_event_ptr;
    }

    const Tt& period()const{
        return _period;
    }

    const Tt& start()const{
        return _start;
    }

private:
    Tt _period;
    Tt _start;

};





template<class Tt, class Ty>
class PyStopEvent : public PyAnyEvent<Tt, Ty>{

public:
    PyStopEvent(py::str name, py::object when, py::object check_if, py::object mask, py::bool_ hide_mask):PyAnyEvent<Tt, Ty>(name, when, check_if, mask, hide_mask){}

    AnyEvent<Tt, Ty>* toEvent(const _Shape& shape, py::tuple args) override{
        delete this->_event_ptr;
        this->_event_ptr = new StopEvent<Tt, Ty>(this->_name, to_event<Tt, Ty>(this->py_when, shape, args), to_event_check<Tt, Ty>(this->py_check_if, shape, args), to_Func<Tt, Ty>(this->py_mask, shape, args), this->hide_mask);
        return this->_event_ptr;
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

    PyODE(py::object f, const Tt t0, const py::array q0, const Tt rtol, const Tt atol, const Tt min_step, const Tt max_step, const Tt first_step, const py::tuple args, const py::str method, py::object events, py::object mask, py::str savedir, const bool save_events_only) : ODE<Tt, Ty>(to_Func<Tt, Ty>(f, shape(q0), args), t0, toCPP_Array<Tt, Ty>(q0), rtol, atol, min_step, max_step, first_step, {}, method.cast<std::string>(), to_Events<Tt, Ty>(events, shape(q0), args), to_Func<Tt, Ty>(mask, shape(q0), args), savedir.cast<std::string>(), save_events_only), q0_shape(shape(q0)){}

    PyODE(const Func<Tt, Ty> f, const Tt t0, const Ty q0, const Tt rtol, const Tt atol, const Tt min_step, const Tt max_step, const Tt first_step, const std::vector<Tt> args = {}, const std::string& method = "RK45", const std::vector<AnyEvent<Tt, Ty>*>& events = {}, const Func<Tt, Ty> mask=nullptr, const std::string& savedir = "", const bool& save_events_only=false) : ODE<Tt, Ty>(f, t0, q0, rtol, atol, min_step, max_step, first_step, args, method, events, mask, savedir, save_events_only), q0_shape({q0.size()}){}

    PyODE(PyODE<Tt, Ty>&& other):ODE<Tt, Ty>(std::move(other)), q0_shape(other.q0_shape){}

    PyODE(const PyODE<Tt, Ty>& other) : ODE<Tt, Ty>(other), q0_shape(other.q0_shape){}

    PySolverState<Tt, Ty> py_state() const{
        SolverState<Tt, Ty> s = this->state();
        return PySolverState<Tt, Ty>(s.t, s.q, s.habs, s.event, s.diverges, s.is_stiff, s.is_running, s.is_dead, s.N, s.message, q0_shape); 
    }

    PyOdeResult<Tt, Ty> py_integrate(const Tt& interval, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        OdeResult<Tt, Ty> res = this->integrate(interval, max_frames, max_events, terminate, max_prints, include_first);
        PyOdeResult<Tt, Ty> py_res = PyOdeResult<Tt, Ty>(res, this->q0_shape);
        return py_res;
    }

    PyOdeResult<Tt, Ty> py_go_to(const Tt& t, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        return this->py_integrate(t, max_frames, max_events, terminate, max_prints, include_first);
    }

    py::array_t<Tt> t_array()const{
        py::array_t<Tt> res = to_numpy<Tt>(this->t());
        return res;
    }

    py::array_t<Tt> q_array()const{
        py::array_t<Tt> res = to_numpy<Tt>(flatten<Tt, Ty>(this->q()), getShape(this->t().size(), this->q0_shape));
        return res;
    }

    const _Shape q0_shape;
};

#pragma GCC visibility pop


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/

template<class Tt, class Ty>
Func<Tt, Ty> to_Func(py::object f, const _Shape& shape, py::tuple py_args){
    if (f.is_none()){
        return nullptr;
    }

    Func<Tt, Ty> g;
    if (py_args.empty()){
        g = [f, shape](const Tt& t, const Ty& y, const std::vector<Tt>& args) -> Ty {
            return fast_convert<Tt, Ty>(f(t, to_numpy<Tt>(y, shape), *to_tuple(args)));
        };
    }
    else{
        g = [f, shape, py_args](const Tt& t, const Ty& y, const std::vector<Tt>& _) -> Ty {
            Ty res = fast_convert<Tt, Ty>(f(t, to_numpy<Tt>(y, shape), *py_args));
            return res;
        };
    }

    return g;
}

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(py::object py_event, const _Shape& shape, py::tuple py_args){
    if (py_event.is_none()){
        return nullptr;
    }
    event_f<Tt, Ty> g;
    if (py_args.empty()){
        g = [py_event, shape](const Tt& t, const Ty& f, const std::vector<Tt>& args) -> Tt {
            return py_event(t, to_numpy<Tt>(f, shape), *to_tuple(args)).template cast<Tt>();
        };
    }
    else{
        g = [py_event, shape, py_args](const Tt& t, const Ty& f, const std::vector<Tt>& _) -> Tt {
            return py_event(t, to_numpy<Tt>(f, shape), *py_args).template cast<Tt>();
        };
    }

    return g;
}

template<class Tt, class Ty>
is_event_f<Tt, Ty> to_event_check(py::object py_event_check, const _Shape& shape, py::tuple py_args){
    if (py_event_check.is_none()){
        return nullptr;
    }

    is_event_f<Tt, Ty> g;
    if (py_args.empty()){
        g = [py_event_check, shape](const Tt& t, const Ty& f, const std::vector<Tt>& args) -> bool {
            return py_event_check(t, to_numpy<Tt>(f, shape), *to_tuple(args)).equal(py::bool_(true));
        };
    }
    else{
        g = [py_event_check, shape, py_args](const Tt& t, const Ty& f, const std::vector<Tt>& _) -> bool {
            return py_event_check(t, to_numpy<Tt>(f, shape), *py_args).equal(py::bool_(true));
        };
    }

    return g;
}

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A) {
    // Convert A to a numpy array of type Tt, ensuring proper type conversion
    py::array_t<Tt> converted_A = py::array_t<Tt>(A);

    size_t n = converted_A.size();
    Ty res(n);

    const Tt* data = static_cast<const Tt*>(converted_A.data());

    for (size_t i = 0; i < n; i++) {
        res[i] = data[i];
    }
    return res;
}


template<class Tt, class Ty>
Ty fast_convert(const py::array_t<Tt>& A){
    size_t n = A.size();
    Ty res(1, n);
    const Tt* data = static_cast<const Tt*>(A.data());
    for (size_t i=0; i<n; i++){
        res(0, i) = data[i];
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
py::array_t<Scalar> to_numpy(const ArrayType& array, const _Shape& q0_shape){
    if (q0_shape.size() == 0){
        py::array_t<Scalar> res(array.size(), array.data());
        return res;
    }
    else{
        py::array_t<Scalar> res(q0_shape, array.data());
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
std::vector<AnyEvent<Tt, Ty>*> to_Events(py::object events, const _Shape& shape, py::tuple args){   
    if (events.is_none()){
        return {};
    }
    std::vector<AnyEvent<Tt, Ty>*> res;
    for (const py::handle& item : events){
        res.push_back(item.cast<PyAnyEvent<Tt, Ty>&>().toEvent(shape, args));
    }
    return res;
}

template<class Tt, class Ty>
void define_ode_module(py::module& m) {

    py::class_<PyAnyEvent<Tt, Ty>>(m, "AnyEvent", py::module_local())
        .def_property_readonly("name", [](const PyAnyEvent<Tt, Ty>& self){
            return self.name();
        })
        .def_property_readonly("mask", [](const PyAnyEvent<Tt, Ty>& self){
            return self.py_mask;
        })
        .def_property_readonly("hide_mask", [](const PyAnyEvent<Tt, Ty>& self){
            return self.hide_mask;
        });


    py::class_<PyEvent<Tt, Ty>, PyAnyEvent<Tt, Ty>>(m, "Event", py::module_local())
        .def(py::init<py::str, py::object, py::object, py::object, py::bool_, Tt>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("check_if")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12)
        .def_property_readonly("when", [](const PyEvent<Tt, Ty>& self){
            return self.py_when;})
        .def_property_readonly("check_if", [](const PyEvent<Tt, Ty>& self){
            return self.py_check_if;});

        py::class_<PyPerEvent<Tt, Ty>, PyAnyEvent<Tt, Ty>>(m, "PeriodicEvent", py::module_local())
            .def(py::init<py::str, Tt, Tt, py::object, py::bool_>(),
                py::arg("name"),
                py::arg("period"),
                py::arg("start")=0,
                py::arg("mask")=py::none(),
                py::arg("hide_mask")=false)
            .def_property_readonly("period", [](const PyPerEvent<Tt, Ty>& self){
                    return self.period();})
            .def_property_readonly("start", [](const PyPerEvent<Tt, Ty>& self){
                return self.start();});

        py::class_<PyStopEvent<Tt, Ty>, PyAnyEvent<Tt, Ty>>(m, "StopEvent", py::module_local())
            .def(py::init<py::str, py::object, py::object, py::object, py::bool_>(),
                py::arg("name"),
                py::arg("when"),
                py::arg("check_if")=py::none(),
                py::arg("mask")=py::none(),
                py::arg("hide_mask")=false)
            .def_property_readonly("when", [](const PyStopEvent<Tt, Ty>& self){
                return self.py_when;})
            .def_property_readonly("check_if", [](const PyStopEvent<Tt, Ty>& self){
                return self.py_check_if;});

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
        .def(py::init<py::object, Tt, py::array, Tt, Tt, Tt, Tt, Tt, py::tuple, py::str, py::object, py::object, py::str, py::bool_>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<Tt>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("events")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("savedir")="",
            py::arg("save_events_only")=false)
        .def("integrate", &PyODE<Tt, Ty>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("max_events")=-1,
            py::arg("terminate")=true,
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("go_to", &PyODE<Tt, Ty>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("max_frames")=-1,
        py::arg("max_events")=-1,
        py::arg("terminate")=true,
        py::arg("max_prints")=0,
        py::arg("include_first")=false)
        .def("copy", [](const PyODE<Tt, Ty>& self){return PyODE<Tt, Ty>(self);})
        .def("advance", [](PyODE<Tt, Ty>& self){return self.advance();})
        .def("resume", [](PyODE<Tt, Ty>& self){return self.resume();})
        .def("free", [](PyODE<Tt, Ty>& self){return self.free();})
        .def("state", &PyODE<Tt, Ty>::py_state)
        .def("save_data", [](PyODE<Tt, Ty>& self, py::str savedir){return self.save_data(savedir.cast<std::string>());}, py::arg("savedir"))
        .def("clear", &PyODE<Tt, Ty>::clear)
        .def_property_readonly("t", &PyODE<Tt, Ty>::t_array)
        .def_property_readonly("q", &PyODE<Tt, Ty>::q_array)
        .def_property_readonly("event_map", [](const PyODE<Tt, Ty>& self){return to_PyDict(self.event_map());})
        .def_property_readonly("solver_filename", [](const PyODE<Tt, Ty>& self){return py::str(self.solver_filename());})
        .def_property_readonly("runtime", &PyODE<Tt, Ty>::runtime)
        .def_property_readonly("is_stiff", &PyODE<Tt, Ty>::is_stiff)
        .def_property_readonly("diverges", &PyODE<Tt, Ty>::diverges)
        .def_property_readonly("is_dead", &PyODE<Tt, Ty>::is_dead);


    m.def("integrate_all", [](py::object list, const Tt& interval, const int& max_frames, const int& max_events, const bool& terminate, const int& threads, const int& max_prints){
        std::vector<ODE<Tt, Ty>*> array;
        for (const py::handle& item : list){
            array.push_back(&(item.cast<PyODE<Tt, Ty>&>()));
        }
        integrate_all(array, interval, max_frames, max_events, terminate, threads, max_prints);
    }, py::arg("ode_array"), py::arg("interval"), py::arg("max_frames")=-1, py::arg("max_events")=-1, py::arg("terminate")=true, py::arg("threads")=-1, py::arg("max_prints")=0);
}


//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

#endif
