#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include "ode.hpp"

namespace py = pybind11;

//assumes empty args given as std::vector, use only for ODE instanciated from python directly
template<class T, int N>
Func<T, N> to_Func(py::object f, const _Shape& shape, py::tuple args=py::tuple());

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

template<class T, int N>
event_f<T, N> to_event(py::object py_event, const _Shape& shape, py::tuple args);

template<class T, int N>
is_event_f<T, N> to_event_check(py::object py_event_check, const _Shape& shape, py::tuple py_args);

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map);

template<class T, class Ty>
Ty toCPP_Array(const py::array& A);

template<class T, int N>
vec<T, N> fast_convert(const py::array_t<T>& A);

template<class T, int N>
std::vector<T> flatten(const std::vector<vec<T, N>>&);

template<class T>
py::tuple to_tuple(const std::vector<T>& vec);

template<class T, int N>
std::vector<Event<T, N>*> to_Events(py::object events, const _Shape& shape, py::tuple args);


template<typename T>
T open_capsule(py::capsule f){
    void* ptr = f.get_pointer();
    if (ptr == nullptr){
        return nullptr;
    }
    else{
        return reinterpret_cast<T>(ptr);
    }
}


#pragma GCC visibility push(hidden)


template<class T, int N>
struct OdeFunc{

    vec<T, N>(*f)(const T&, const vec<T, N>&, const std::vector<T>&);
};




template<class T, int N>
class PyEvent{

public:
    PyEvent(py::str name, py::object when, py::object check_if, py::object mask, py::bool_ hide_mask, T event_tol):_name(name.cast<std::string>()), py_when(when), py_check_if(check_if), py_mask(mask), hide_mask(hide_mask), _event_tol(event_tol){}

    virtual PreciseEvent<T, N>* toEvent(const _Shape& shape, py::tuple args=py::tuple()) {
        delete this->_event_ptr;
        this->_event_ptr = new PreciseEvent<T, N>(this->_name, to_event<T, N>(this->py_when, shape, args), to_event_check<T, N>(this->py_check_if, shape, args), to_Func<T, N>(this->py_mask, shape, args), this->hide_mask, this->_event_tol);
        return this->_event_ptr;
    }

    py::str name()const{
        return py::str(_name);
    }

    virtual ~PyEvent(){
        delete this->_event_ptr;
        this->_event_ptr = nullptr;
    }

    std::string _name;

    py::object py_when;
    py::object py_check_if;
    py::object py_mask;
    bool hide_mask;
    
    PreciseEvent<T, N>* _event_ptr = nullptr;

    T _event_tol;

};


template<class T, int N>
class PyPerEvent : public PyEvent<T, N>{

public:
    PyPerEvent(py::str name, const T& period, const T& start, py::object mask, py::bool_ hide_mask):PyEvent<T, N>(name, py::none(), py::none(), mask, hide_mask, 0), _period(period), _start(start){}

    PreciseEvent<T, N>* toEvent(const _Shape& shape, py::tuple args) override{
        delete this->_event_ptr;
        this->_event_ptr = new PeriodicEvent<T, N>(this->_name, this->_period, this->_start, to_Func<T, N>(this->py_mask, shape, args), this->hide_mask);
        return this->_event_ptr;
    }

    const T& period()const{
        return _period;
    }

    const T& start()const{
        return _start;
    }

private:
    T _period;
    T _start;

};


template<class T, int N>
class PySolverState : public SolverState<T, N>{

public:
    PySolverState(const T& t, const vec<T, N>& q, const T& habs, const std::string& event, const bool& diverges, const bool& is_stiff, const bool& is_running, const bool& is_dead, const size_t& Nt, const std::string& message, const _Shape& shape): SolverState<T, N>(t, q, habs, event, diverges, is_stiff, is_running, is_dead, Nt, message), shape(shape) {}

    const std::vector<size_t> shape;

};


template<class T, int N>
struct PyOdeResult{

    PyOdeResult(const OdeResult<T, N>& r, const _Shape& q0_shape): res(r), q0_shape(q0_shape){}

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = this->res.t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        std::vector<vec<T, N>> q_data = this->res.q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(flatten<T, N>(q_data), getShape(q_data.size(), this->q0_shape));
        return py::make_tuple(t_res, q_res);
    }

    const OdeResult<T, N> res;
    const _Shape& q0_shape;

};



class PyOdeBase {
public:

    PyOdeResult<double, -1>* integrate_none(const double& interval, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        none();
        return nullptr;
    }

    PyOdeResult<double, -1>* go_to_none(const double& t, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        none();
        return nullptr;
    }

    py::none save_data_none(py::str savedir) const {
        return none();
    }

    py::tuple event_data_none(py::str event) const{
        return none();
    }


    py::object none() const {
        PyErr_SetString(PyExc_NotImplementedError, "Method not implemented.");
        throw py::error_already_set();
    }

/*
Empty class to expose to python.
Although this is empty, python methods will be exposed
whose only purpose is to demonstrate the all methods the derived classes have.

Its purpose is to not be instanciated, so that a class can inherit from its general behavior without calling super().__init__,
as long as the methods are bypassed.

The only reason this is necessary is so that another PyODE (see class below) object can be passed into any class "A" that inherits from this abstract one,
and by overrding __getattr__, the new object "A" will behave like the one passed into it.
This way, one can inherit from PyODE for any compiled template parameter which would otherwise be impossible
since there is no concrete exposed class implementation for stack allocated arrays, but objects of it are returned for the system size
provided.
*/

};

template<class T, int N>
class PyODE : public ODE<T, N>, public PyOdeBase{
public:

    PyODE(const py::object& f, const T t0, const py::array q0, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const py::str method, py::object events, py::object mask, py::str savedir, const bool save_events_only) : ODE<T, N>(to_Func<T, N>(f, shape(q0), args), t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, {}, method.cast<std::string>(), to_Events<T, N>(events, shape(q0), args), to_Func<T, N>(mask, shape(q0), args), savedir.cast<std::string>(), save_events_only), q0_shape(shape(q0)){}

    PyODE(const Func<T, N> f, const T t0, const vec<T, N> q0, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const std::vector<T> args = {}, const std::string& method = "RK45", const std::vector<Event<T, N>*>& events = {}, const Func<T, N> mask=nullptr, const std::string& savedir = "", const bool& save_events_only=false) : ODE<T, N>(f, t0, q0, rtol, atol, min_step, max_step, first_step, args, method, events, mask, savedir, save_events_only), q0_shape({static_cast<size_t>(q0.size())}){}

    PyODE(PyODE<T, N>&& other):ODE<T, N>(std::move(other)), q0_shape(other.q0_shape){}

    PyODE(const PyODE<T, N>& other) : ODE<T, N>(other), q0_shape(other.q0_shape){}

    PySolverState<T, N> py_state() const{
        SolverState<T, N> s = this->state();
        return PySolverState<T, N>(s.t, s.q, s.habs, s.event, s.diverges, s.is_stiff, s.is_running, s.is_dead, s.Nt, s.message, q0_shape); 
    }

    PyOdeResult<T, N> py_integrate(const T& interval, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        OdeResult<T, N> res = this->integrate(interval, max_frames, max_events, terminate, max_prints, include_first);
        PyOdeResult<T, N> py_res = PyOdeResult<T, N>(res, this->q0_shape);
        return py_res;
    }

    PyOdeResult<T, N> py_go_to(const T& t, const int max_frames, const int& max_events, const bool& terminate, const int& max_prints, const bool& include_first){
        return this->py_integrate(t, max_frames, max_events, terminate, max_prints, include_first);
    }

    py::array_t<T> t_array()const{
        py::array_t<T> res = to_numpy<T>(this->t());
        return res;
    }

    py::array_t<T> q_array()const{
        py::array_t<T> res = to_numpy<T>(flatten<T, N>(this->q()), getShape(this->t().size(), this->q0_shape));
        return res;
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = this->t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        std::vector<vec<T, N>> q_data = this->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(flatten<T, N>(q_data), getShape(q_data.size(), this->q0_shape));
        return py::make_tuple(t_res, q_res);
    }

    const _Shape q0_shape;
};

template<class T, int N>
class DynamicPyOde : public PyODE<T, N>{


public:
    DynamicPyOde(const py::capsule& f, const T t0, const py::array q0, const py::capsule& mask, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const py::str method, py::capsule events, py::str savedir, const bool save_events_only) : PyODE<T, N>(open_capsule<Fptr<T, N>>(f), t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<T, std::vector<T>>(args), method.cast<std::string>(), *open_capsule<std::vector<Event<T, N>*>*>(events), open_capsule<Fptr<T, N>>(mask), savedir.cast<std::string>(), save_events_only){}

    DynamicPyOde(const py::capsule& f, const T t0, const py::array q0, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const py::str method, py::capsule events, py::str savedir, const bool save_events_only) : PyODE<T, N>(open_capsule<Fptr<T, N>>(f), t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<T, std::vector<T>>(args), method.cast<std::string>(), *open_capsule<std::vector<Event<T, N>*>*>(events), nullptr, savedir.cast<std::string>(), save_events_only){}
};


#pragma GCC visibility pop


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/

template<class T, int N>
Func<T, N> to_Func(py::object f, const _Shape& shape, py::tuple py_args){
    if (f.is_none()){
        return nullptr;
    }

    Func<T, N> g;
    if (py_args.empty()){
        g = [f, shape](const T& t, const vec<T, N>& y, const std::vector<T>& args) -> vec<T, N> {
            return fast_convert<T, N>(f(t, to_numpy<T>(y, shape), *to_tuple(args)));
        };
    }
    else{
        g = [f, shape, py_args](const T& t, const vec<T, N>& y, const std::vector<T>& _) -> vec<T, N> {
            vec<T, N> res = fast_convert<T, N>(f(t, to_numpy<T>(y, shape), *py_args));
            return res;
        };
    }

    return g;
}

template<class T, int N>
event_f<T, N> to_event(py::object py_event, const _Shape& shape, py::tuple py_args){
    if (py_event.is_none()){
        return nullptr;
    }
    event_f<T, N> g;
    if (py_args.empty()){
        g = [py_event, shape](const T& t, const vec<T, N>& f, const std::vector<T>& args) -> T {
            return py_event(t, to_numpy<T>(f, shape), *to_tuple(args)).template cast<T>();
        };
    }
    else{
        g = [py_event, shape, py_args](const T& t, const vec<T, N>& f, const std::vector<T>& _) -> T {
            return py_event(t, to_numpy<T>(f, shape), *py_args).template cast<T>();
        };
    }

    return g;
}

template<class T, int N>
is_event_f<T, N> to_event_check(py::object py_event_check, const _Shape& shape, py::tuple py_args){
    if (py_event_check.is_none()){
        return nullptr;
    }

    is_event_f<T, N> g;
    if (py_args.empty()){
        g = [py_event_check, shape](const T& t, const vec<T, N>& f, const std::vector<T>& args) -> bool {
            return py_event_check(t, to_numpy<T>(f, shape), *to_tuple(args)).equal(py::bool_(true));
        };
    }
    else{
        g = [py_event_check, shape, py_args](const T& t, const vec<T, N>& f, const std::vector<T>& _) -> bool {
            return py_event_check(t, to_numpy<T>(f, shape), *py_args).equal(py::bool_(true));
        };
    }

    return g;
}

template<class T, class Ty>
Ty toCPP_Array(const py::array& A) {
    py::array_t<T> converted_A = py::array_t<T>(A);

    size_t n = converted_A.size();
    Ty res(n);

    const T* data = static_cast<const T*>(converted_A.data());

    for (size_t i = 0; i < n; i++) {
        res[i] = data[i];
    }
    return res;
}


template<class T, int N>
vec<T, N> fast_convert(const py::array_t<T>& A){
    size_t n = A.size();
    vec<T, N> res(1, n);
    const T* data = static_cast<const T*>(A.data());
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

template<class T, int N>
std::vector<T> flatten(const std::vector<vec<T, N>>& f){
    size_t nt = f.size();
    if (nt == 0){
        return {};
    }
    size_t nd = f[0].size();
    std::vector<T> res(nt*nd);

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

template<class T, int N>
std::vector<Event<T, N>*> to_Events(py::object events, const _Shape& shape, py::tuple args){   
    if (events.is_none()){
        return {};
    }
    std::vector<Event<T, N>*> res;
    for (const py::handle& item : events){
        res.push_back(item.cast<PyEvent<T, N>&>().toEvent(shape, args));
    }
    return res;
}

template<class T, int N>
void define_ode_module(py::module& m) {

    py::class_<PyEvent<T, N>>(m, "Event", py::module_local())
        .def(py::init<py::str, py::object, py::object, py::object, py::bool_, T>(),
        py::arg("name"),
        py::arg("when"),
        py::arg("check_if")=py::none(),
        py::arg("mask")=py::none(),
        py::arg("hide_mask")=false,
        py::arg("event_tol")=1e-12)
        .def_property_readonly("name", [](const PyEvent<T, N>& self){
            return self.name();
        })
        .def_property_readonly("mask", [](const PyEvent<T, N>& self){
            return self.py_mask;
        })
        .def_property_readonly("hide_mask", [](const PyEvent<T, N>& self){
            return self.hide_mask;
        })
        .def_property_readonly("when", [](const PyEvent<T, N>& self){
            return self.py_when;})
        .def_property_readonly("check_if", [](const PyEvent<T, N>& self){
            return self.py_check_if;});
        
    py::class_<PyPerEvent<T, N>, PyEvent<T, N>>(m, "PeriodicEvent", py::module_local())
        .def(py::init<py::str, T, T, py::object, py::bool_>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("start")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false)
        .def_property_readonly("period", [](const PyPerEvent<T, N>& self){
                return self.period();})
        .def_property_readonly("start", [](const PyPerEvent<T, N>& self){
            return self.start();});

    py::class_<PyOdeResult<T, N>>(m, "OdeResult", py::module_local())
        .def_property_readonly("t", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(self.res.t);
        })
        .def_property_readonly("q", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(flatten<T, N>(self.res.q), getShape(self.res.t.size(), self.q0_shape));
        })
        .def_property_readonly("event_map", [](const PyOdeResult<T, N>& self){
            return to_PyDict(self.res.event_map);
        })
        .def("event_data", &PyOdeResult<T, N>::event_data, py::arg("event"))
        .def_property_readonly("diverges", [](const PyOdeResult<T, N>& self){return self.res.diverges;})
        .def_property_readonly("is_stiff", [](const PyOdeResult<T, N>& self){return self.res.is_stiff;})
        .def_property_readonly("success", [](const PyOdeResult<T, N>& self){return self.res.success;})
        .def_property_readonly("runtime", [](const PyOdeResult<T, N>& self){return self.res.runtime;})
        .def_property_readonly("message", [](const PyOdeResult<T, N>& self){return self.res.message;})
        .def("examine", [](const PyOdeResult<T, N>& self){return self.res.examine();});


    py::class_<PySolverState<T, N>>(m, "SolverState", py::module_local())
        .def_property_readonly("t", [](const PySolverState<T, N>& self){return self.t;})
        .def_property_readonly("q", [](const PySolverState<T, N>& self){return to_numpy<T>(self.q, self.shape);})
        .def_property_readonly("event", [](const PySolverState<T, N>& self){return self.event;})
        .def_property_readonly("diverges", [](const PySolverState<T, N>& self){return self.diverges;})
        .def_property_readonly("is_stiff", [](const PySolverState<T, N>& self){return self.is_stiff;})
        .def_property_readonly("is_running", [](const PySolverState<T, N>& self){return self.is_running;})
        .def_property_readonly("is_dead", [](const PySolverState<T, N>& self){return self.is_dead;})
        .def_property_readonly("N", [](const PySolverState<T, N>& self){return self.Nt;})
        .def_property_readonly("message", [](const PySolverState<T, N>& self){return self.message;})
        .def("show", [](const PySolverState<T, N>& self){return self.show();});








    py::class_<PyOdeBase>(m, "ODE", py::module_local())
        .def(py::init<>())
        .def("integrate", &PyOdeBase::integrate_none,
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("max_events")=-1,
            py::arg("terminate")=true,
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("go_to", &PyOdeBase::go_to_none,
            py::arg("t"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("max_events")=-1,
            py::arg("terminate")=true,
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("copy", &PyOdeBase::none)
        .def("advance", &PyOdeBase::none)
        .def("resume", &PyOdeBase::none)
        .def("free", &PyOdeBase::none)
        .def("state", &PyOdeBase::none)
        .def("save_data", &PyOdeBase::save_data_none, py::arg("savedir"))
        .def("clear", &PyOdeBase::none)
        .def("event_data", &PyOdeBase::save_data_none, py::arg("event"))
        .def_property_readonly("dim", &PyOdeBase::none)
        .def_property_readonly("t", &PyOdeBase::none)
        .def_property_readonly("q", &PyOdeBase::none)
        .def_property_readonly("event_map", &PyOdeBase::none)
        .def_property_readonly("solver_filename", &PyOdeBase::none)
        .def_property_readonly("runtime", &PyOdeBase::none)
        .def_property_readonly("is_stiff", &PyOdeBase::none)
        .def_property_readonly("diverges", &PyOdeBase::none)
        .def_property_readonly("is_dead", &PyOdeBase::none);

    py::class_<PyODE<T, N>, PyOdeBase>(m, "LowLevelODE", py::module_local())
        .def(py::init<py::object, T, py::array, T, T, T, T, T, py::tuple, py::str, py::object, py::object, py::str, py::bool_>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("events")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("savedir")="",
            py::arg("save_events_only")=false)
        .def("integrate", &PyODE<T, N>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("max_events")=-1,
            py::arg("terminate")=true,
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("go_to", &PyODE<T, N>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("max_frames")=-1,
        py::arg("max_events")=-1,
        py::arg("terminate")=true,
        py::arg("max_prints")=0,
        py::arg("include_first")=false)
        .def("copy", [](const PyODE<T, N>& self){return PyODE<T, N>(self);})
        .def("advance", [](PyODE<T, N>& self){return self.advance();})
        .def("resume", [](PyODE<T, N>& self){return self.resume();})
        .def("free", [](PyODE<T, N>& self){return self.free();})
        .def("state", &PyODE<T, N>::py_state)
        .def("save_data", [](PyODE<T, N>& self, py::str savedir){return self.save_data(savedir.cast<std::string>());}, py::arg("savedir"))
        .def("clear", &PyODE<T, N>::clear)
        .def("event_data", &PyODE<T, N>::event_data, py::arg("event"))
        .def_property_readonly("dim", [](const PyODE<T, N>& self){return self.solver().Nsys();})
        .def_property_readonly("t", &PyODE<T, N>::t_array)
        .def_property_readonly("q", &PyODE<T, N>::q_array)
        .def_property_readonly("event_map", [](const PyODE<T, N>& self){return to_PyDict(self.event_map());})
        .def_property_readonly("solver_filename", [](const PyODE<T, N>& self){return py::str(self.solver_filename());})
        .def_property_readonly("runtime", &PyODE<T, N>::runtime)
        .def_property_readonly("is_stiff", &PyODE<T, N>::is_stiff)
        .def_property_readonly("diverges", &PyODE<T, N>::diverges)
        .def_property_readonly("is_dead", &PyODE<T, N>::is_dead)
        .def_property_readonly("_ode_obj", [](PyODE<T, N>& self){return &self;});
    
    py::class_<DynamicPyOde<T, N>, PyODE<T, N>>(m, "_DynamicODE", py::module_local())
        .def(py::init<py::capsule, T, py::array, py::capsule, T, T, T, T, T, py::tuple, py::str, py::capsule, py::str, py::bool_>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::arg("mask"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=inf<T>(),
        py::arg("first_step")=0.,
        py::arg("args")=py::tuple(),
        py::arg("method")="RK45",
        py::arg("events")=py::none(),
        py::arg("savedir")="",
        py::arg("save_events_only")=false)

        .def(py::init<py::capsule, T, py::array, T, T, T, T, T, py::tuple, py::str, py::capsule, py::str, py::bool_>(),
        py::arg("f"),
        py::arg("t0"),
        py::arg("q0"),
        py::kw_only(),
        py::arg("rtol")=1e-6,
        py::arg("atol")=1e-12,
        py::arg("min_step")=0.,
        py::arg("max_step")=inf<T>(),
        py::arg("first_step")=0.,
        py::arg("args")=py::tuple(),
        py::arg("method")="RK45",
        py::arg("events")=py::none(),
        py::arg("savedir")="",
        py::arg("save_events_only")=false);

    m.def("integrate_all", [](py::object list, const T& interval, const int& max_frames, const int& max_events, const bool& terminate, const int& threads, const int& max_prints){
            std::vector<ODE<T, N>*> array;
            for (const py::handle& item : list) {
                array.push_back(&item.attr("_ode_obj").cast<PyODE<T, N>&>());
            }
            integrate_all(array, interval, max_frames, max_events, terminate, threads, max_prints);
        }, py::arg("ode_array"), py::arg("interval"), py::arg("max_frames")=-1, py::arg("max_events")=-1, py::arg("terminate")=true, py::arg("threads")=-1, py::arg("max_prints")=0);
        
}


//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

#endif
