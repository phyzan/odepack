#ifndef PYODE_HPP
#define PYODE_HPP


#include <cstddef>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "variational.hpp"
#include "solvers.hpp"

namespace py = pybind11;

//assumes empty args given as std::vector, use only for ODE instanciated from python directly
template<typename T, int N>
Func<T, N> to_Func(py::object f, const _Shape& shape, py::tuple args=py::tuple());

_Shape shape(const py::object& arr);

_Shape getShape(const size_t& dim1, const _Shape& shape){
    std::vector<size_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const _Shape& q0_shape={});

template<typename T, int N>
event_f<T, N> to_event(py::object py_event, const _Shape& shape, py::tuple args);

template<typename T, int N>
is_event_f<T, N> to_event_check(py::object py_event_check, const _Shape& shape, py::tuple py_args);

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map);

template<typename T, class Ty>
Ty toCPP_Array(const py::object& obj);

template<typename T, int N>
vec<T, N> fast_convert(const py::array_t<T>& A);

template<typename T, int N>
std::vector<T> flatten(const std::vector<vec<T, N>>&);

template<typename T>
py::tuple to_tuple(const std::vector<T>& vec);

template<typename T, int N>
std::vector<Event<T, N>*> to_Events(py::object events, const _Shape& shape, py::tuple args);

std::vector<EventOptions> to_Options(const py::dict& d) {
    std::vector<EventOptions> result;

    for (auto item : d) {
        std::string key = py::cast<std::string>(item.first);
        auto value = item.second;

        EventOptions opt;
        opt.name = key;

        if (py::isinstance<py::int_>(value)) {
            opt.max_events = value.cast<int>();
        } else if (py::isinstance<py::tuple>(value)) {
            auto tup = value.cast<py::tuple>();
            opt.max_events = tup[0].cast<int>();
            if (tup.size() > 1){
                opt.terminate = tup[1].cast<bool>();
            }
            if (tup.size() > 2){
                opt.period = tup[2].cast<int>();
            }
            if (tup.size() > 3){
                throw py::value_error("Tuple size in event options dict value must be at most 3");
            }
            
        } else {
            throw py::value_error("Expected int or (int, bool, int) tuple in event options dict value");
        }

        result.push_back(opt);
    }
    return result;
}



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


template<typename T, int N>
class PyEvent{

public:
    PyEvent(py::str name, py::object when, py::object check_if, py::object mask, py::bool_ hide_mask, T event_tol):_name(name.cast<std::string>()), py_when(when), py_check_if(check_if), py_mask(mask), hide_mask(hide_mask), _event_tol(event_tol){}

    virtual PreciseEvent<T, N>* toEvent(const _Shape& shape, py::tuple args=py::tuple()) {
        return new PreciseEvent<T, N>(this->_name, to_event<T, N>(this->py_when, shape, args), to_event_check<T, N>(this->py_check_if, shape, args), to_Func<T, N>(this->py_mask, shape, args), this->hide_mask, this->_event_tol);
    }

    py::str name()const{
        return py::str(_name);
    }

    virtual ~PyEvent(){}

    std::string _name;

    py::object py_when;
    py::object py_check_if;
    py::object py_mask;
    bool hide_mask;

    T _event_tol;

};


template<typename T, int N>
class PyPerEvent : public PyEvent<T, N>{

public:
    PyPerEvent(py::str name, const T& period, const T& start, py::object mask, py::bool_ hide_mask):PyEvent<T, N>(name, py::none(), py::none(), mask, hide_mask, 0), _period(period), _start(start){}

    PyPerEvent(py::str name, const T& period, py::object mask, py::bool_ hide_mask):PyPerEvent<T, N>(name, period, inf<T>(), mask, hide_mask){}

    PreciseEvent<T, N>* toEvent(const _Shape& shape, py::tuple args) override{
        return new PeriodicEvent<T, N>(this->_name, this->_period, this->_start, to_Func<T, N>(this->py_mask, shape, args), this->hide_mask);
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


template<typename T, int N>
struct PyEventWrapper{

    PyEventWrapper(const py::capsule& obj, size_t qsize, size_t args_size) : events(open_capsule<std::vector<Event<T, N>*>*>(obj)), q_size(qsize), args_size(args_size){}

    std::vector<Event<T, N>*>* events;
    size_t q_size;
    size_t args_size;

};


template<typename T, int N>
struct PyFuncWrapper{

    Functor<T, N> rhs;
    size_t q_size;
    size_t args_size;

    PyFuncWrapper(const py::capsule& obj, size_t qsize, size_t args_size) : rhs(open_capsule<Fvoidptr<T, N>>(obj)), q_size(qsize), args_size(args_size) {}

    py::array_t<T> call(const T& t, const py::object& py_q, const std::vector<T>& args) const {
        vec<T, N> q = toCPP_Array<T, vec<T, N>>(py_q);
        vec<T, N> res(q_size);
        if (static_cast<size_t>(q.size()) != q_size || args.size() != args_size){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        this->rhs(res, t, q, args);
        return to_numpy<T>(res);
    }
};





template<typename T, int N>
class PySolverState : public SolverState<T, N>{

public:
    PySolverState(const T& t, const vec<T, N>& q, const T& habs, const std::string& event, const bool& diverges, const bool& is_running, const bool& is_dead, const size_t& Nt, const std::string& message, const _Shape& shape): SolverState<T, N>(t, q, habs, event, diverges, is_running, is_dead, Nt, message), shape(shape) {}

    const std::vector<size_t> shape;

};


template<typename T, int N>
struct PyOdeResult{

    PyOdeResult(const OdeResult<T, N>& r, const _Shape& q0_shape): res(r.clone()), q0_shape(q0_shape){}

    PyOdeResult(const PyOdeResult& other) : res(other.res->clone()), q0_shape(other.q0_shape) {}

    PyOdeResult(PyOdeResult&& other) : res(other.res), q0_shape(std::move(other.q0_shape)) {
        other.res = nullptr;
    }

    virtual ~PyOdeResult(){delete res; res = nullptr;}

    PyOdeResult& operator=(const PyOdeResult& other){
        if (&other != this){
            delete res;
            res = other.res->clone();
            q0_shape = other.q0_shape;
        }
        return *this;
    }

    PyOdeResult& operator=(PyOdeResult&& other){
        if (&other != this){
            delete res;
            res = other.res;
            q0_shape = other.q0_shape;
            other.res = nullptr;
        }
        return *this;
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = this->res->t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        std::vector<vec<T, N>> q_data = this->res->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(flatten<T, N>(q_data), getShape(q_data.size(), this->q0_shape));
        return py::make_tuple(t_res, q_res);
    }

    OdeResult<T, N>* res;
    _Shape q0_shape;

};

template<typename T, int N>
struct PyOdeSolution : public PyOdeResult<T, N>{

    PyOdeSolution(const OdeSolution<T, N>& res, const _Shape& q0_shape) : PyOdeResult<T, N>(res, q0_shape) {
        nd = 1;
        for (size_t i=0; i<q0_shape.size(); i++){
            nd *= q0_shape[i];
        }
    }

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    inline const OdeSolution<T, N>& sol() const{
        return static_cast<const OdeSolution<T, N>&>(*this->res);
    }

    inline py::array_t<T> operator()(const T& t) const{
        return to_numpy<T>(this->sol()(t), this->q0_shape);
    }

    py::array_t<T> operator()(const py::array_t<T>& array) const{
        const size_t nt = array.size();
        _Shape final_shape(array.shape(), array.shape()+array.ndim());
        final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());

        vec<T, N> q(nd);
        const T* data = array.data();
        std::vector<T> res(nt*nd);
        for (size_t i=0; i<nt; i++){
            q = this->sol()(data[i]);
            for (size_t j=0; j<nd; j++){
                res[i*nd+j] = q[j];
            }
        }
        py::array_t<T> r = to_numpy<T>(res, final_shape);
        return r;
    }

    size_t nd;

};



template<typename T, int N>
class PyODE{

public:


    PyODE(const py::object& f, const T t0, const py::object& q0, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, py::object events, const py::str method){
        std::unique_ptr<OdeSolver<T, N>> solver = get_solver<T, N>(method.cast<std::string>(), to_Func<T, N>(f, shape(q0), args), t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<T, std::vector<T>>(args), to_Events<T, N>(events, shape(q0), args));
        ode = new ODE<T, N>(*solver);
        q0_shape = shape(q0);
    }

    PyODE(const PyFuncWrapper<T, N>& func_wrap, const T t0, const py::object& q0, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const PyEventWrapper<T, N>* ev_wrap, const py::str method){
        std::unique_ptr<OdeSolver<T, N>> solver = get_solver<T, N>(method.cast<std::string>(), func_wrap.rhs, t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<T, std::vector<T>>(args), ((ev_wrap != nullptr) ? *ev_wrap->events : std::vector<Event<T, N>*>(0)));
        ode = new ODE<T, N>(*solver);
        q0_shape = {static_cast<size_t>(ode->q()[0].size())};
        _assert_sizes(func_wrap, ev_wrap);
    }

    PyODE(ODE_CONSTRUCTOR(T, N)){
        ode = new ODE<T, N>(get_solver<T, N>(method, ARGS));
        q0_shape = {static_cast<size_t>(ode->q()[0].size())};
    }

    void _assert_sizes(const PyFuncWrapper<T, N>& func_wrap, const PyEventWrapper<T, N>* ev_wrap){
        if (func_wrap.q_size != q0_shape[0] || (ev_wrap != nullptr && ev_wrap->q_size != q0_shape[0])){
            throw py::value_error("Initial conditions, ode function input array and event input array must all have the same size");
        }


        if (func_wrap.args_size != ode->solver().args().size() || ( ev_wrap != nullptr && ev_wrap->args_size != ode->solver().args().size())){
            throw py::value_error("Incompatible args array sizes between ode function, event input args, and given args");
        }
    }

protected:
    PyODE(){}//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE<T, N>& other) : ode(other.ode->clone()), q0_shape(other.q0_shape){}

    PyODE(PyODE<T, N>&& other):ode(other.ode), q0_shape(other.q0_shape){
        other.ode = nullptr;
    }

    PyODE<T, N>& operator=(const PyODE<T, N>& other){
        if (&other == this){
            return *this;
        }
        delete ode;
        ode = other.ode->clone();
        q0_shape = other.q0_shape;
        return *this;
    }

    PyODE<T, N>& operator=(PyODE<T, N>&& other){
        if (&other == this){
            return *this;
        }
        ode = other.ode;
        other.ode = nullptr;
        q0_shape = std::move(other.q0_shape);
        return *this;
    }

    virtual ~PyODE(){
        delete ode;
    }

    PySolverState<T, N> py_state() const{
        SolverState<T, N> s = ode->state();
        return PySolverState<T, N>(s.t, s.q, s.habs, s.event, s.diverges, s.is_running, s.is_dead, s.Nt, s.message, q0_shape); 
    }

    PyOdeResult<T, N> py_integrate(const T& interval, const int max_frames, const py::dict& event_options, const int& max_prints, const bool& include_first){
        OdeResult<T, N> res = ode->integrate(interval, max_frames, to_Options(event_options), max_prints, include_first);
        PyOdeResult<T, N> py_res(res, this->q0_shape);
        return py_res;
    }

    PyOdeSolution<T, N> py_rich_integrate(const T& interval, const py::dict& event_options, const int& max_prints){
        OdeSolution<T, N> res = ode->rich_integrate(interval, to_Options(event_options), max_prints);
        PyOdeSolution<T, N> py_res(res, this->q0_shape);
        return py_res;
    }


    PyOdeResult<T, N> py_go_to(const T& t, const int max_frames, const py::dict& event_options, const int& max_prints, const bool& include_first){
        OdeResult<T, N> res = ode->go_to(t, max_frames, to_Options(event_options), max_prints, include_first);
        PyOdeResult<T, N> py_res(res, this->q0_shape);
        return py_res;
    }

    py::array_t<T> t_array()const{
        py::array_t<T> res = to_numpy<T>(ode->t());
        return res;
    }

    py::array_t<T> q_array()const{
        py::array_t<T> res = to_numpy<T>(flatten<T, N>(ode->q()), getShape(ode->t().size(), this->q0_shape));
        return res;
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = ode->t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        std::vector<vec<T, N>> q_data = ode->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(flatten<T, N>(q_data), getShape(q_data.size(), this->q0_shape));
        return py::make_tuple(t_res, q_res);
    }
    ODE<T, N>* ode;
    _Shape q0_shape;
};


template<typename T, int N>
class PyVarODE : public PyODE<T, N>{

public:

    PyVarODE(const py::object& f, const T t0, const py::object& q0, const T& period, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const py::object& events, const py::str method):PyODE<T, N>(){
        
        vec<T, N> q0_ = toCPP_Array<T, vec<T, N>>(q0);
        std::vector<T> args_ = toCPP_Array<T, std::vector<T>>(args);
        this->ode = new VariationalODE<T, N>(to_Func<T, N>(f, shape(q0), args), t0, q0_, period, rtol, atol, min_step, max_step, first_step, args_, to_Events<T, N>(events, shape(q0), args), method.cast<std::string>());
        this->q0_shape = shape(q0);
    }

    PyVarODE(const PyFuncWrapper<T, N>& func_wrap, const T t0, const py::object& q0, const T& period, const T rtol, const T atol, const T min_step, const T max_step, const T first_step, const py::tuple args, const PyEventWrapper<T, N>* ev_wrap, const py::str method):PyODE<T, N>(){

        this->ode = new VariationalODE<T, N>(func_wrap.rhs, t0, toCPP_Array<T, vec<T, N>>(q0), period, rtol, atol, min_step, max_step, first_step, toCPP_Array<T, std::vector<T>>(args), ((ev_wrap != nullptr) ? *ev_wrap->events : std::vector<Event<T, N>*>(0)), method.cast<std::string>());
        this->q0_shape = {static_cast<size_t>(this->ode->q()[0].size())};
        this->_assert_sizes(func_wrap, ev_wrap);
    }

    PyVarODE(ODE_CONSTRUCTOR(T, N)) : PyODE<T, N>() {
        this->ode = new ODE<T, N>(get_solver<T, N>(method, ARGS));
        this->q0_shape = {static_cast<size_t>(this->ode->q()[0].size())};
    }

    DEFAULT_RULE_OF_FOUR(PyVarODE);


    VariationalODE<T, N>& varode(){
        return *static_cast<VariationalODE<T, N>*>(this->ode);
    }

    const VariationalODE<T, N>& varode() const {
        return *static_cast<VariationalODE<T, N>*>(this->ode);
    }

    py::array_t<T> py_t_lyap() const{
        return to_numpy<T>(varode().t_lyap());
    }

    py::array_t<T> py_lyap() const{
        return to_numpy<T>(varode().lyap());
    }

};



#pragma GCC visibility pop


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/

template<typename T, int N>
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

template<typename T, int N>
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

template<typename T, int N>
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


template<typename T, class Ty>
Ty toCPP_Array(const py::object& obj) {
    // Try to ensure the object is a NumPy array of compatible type
    py::array arr = py::array::ensure(obj);
    if (!arr) throw std::invalid_argument("Input cannot be converted to a NumPy array");

    py::array_t<T> converted_A = py::array_t<T>(arr);
    size_t n = converted_A.size();

    Ty res(n);
    const T* data = static_cast<const T*>(converted_A.data());

    for (size_t i = 0; i < n; i++) {
        res[i] = data[i];
    }

    return res;
}



template<typename T, int N>
vec<T, N> fast_convert(const py::array_t<T>& A){
    size_t n = A.size();
    vec<T, N> res(n);
    const T* data = static_cast<const T*>(A.data());
    for (size_t i=0; i<n; i++){
        res[i] = data[i];
    }

    return res;
}

_Shape shape(const py::object& obj) {
    py::array arr = py::array::ensure(obj);
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
        py::array_t<Scalar> res(std::vector<size_t>{static_cast<size_t>(array.size())}, array.data());
        return res;
    }
    else{
        py::array_t<Scalar> res(q0_shape, array.data());
        return res;
    }
}

template<typename T, int N>
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

template<typename T>
py::tuple to_tuple(const std::vector<T>& arr) {
    py::tuple py_tuple(arr.size());  // Create a tuple of the same size as the vector
    for (size_t i = 0; i < arr.size(); ++i) {
        py_tuple[i] = py::float_(arr[i]);  // Convert each double element to py::float_
    }
    return py_tuple;
}

template<typename T, int N>
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

template<typename T, int N>
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
        .def(py::init<py::str, T, py::object, py::bool_>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false)
        .def(py::init<py::str, T, T, py::object, py::bool_>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("start"),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false)
        .def(py::init<py::str, T, py::object, py::bool_>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false)
        .def_property_readonly("period", [](const PyPerEvent<T, N>& self){
                return self.period();})
        .def_property_readonly("start", [](const PyPerEvent<T, N>& self){
            return self.start();});

    py::class_<PyOdeResult<T, N>>(m, "OdeResult", py::module_local())
        .def_property_readonly("t", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(self.res->t());
        })
        .def_property_readonly("q", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(flatten<T, N>(self.res->q()), getShape(self.res->t().size(), self.q0_shape));
        })
        .def_property_readonly("event_map", [](const PyOdeResult<T, N>& self){
            return to_PyDict(self.res->event_map());
        })
        .def("event_data", &PyOdeResult<T, N>::event_data, py::arg("event"))
        .def_property_readonly("diverges", [](const PyOdeResult<T, N>& self){return self.res->diverges();})
        .def_property_readonly("success", [](const PyOdeResult<T, N>& self){return self.res->success();})
        .def_property_readonly("runtime", [](const PyOdeResult<T, N>& self){return self.res->runtime();})
        .def_property_readonly("message", [](const PyOdeResult<T, N>& self){return self.res->message();})
        .def("examine", [](const PyOdeResult<T, N>& self){return self.res->examine();});
    
    py::class_<PyOdeSolution<T, N>, PyOdeResult<T, N>>(m, "OdeSolution", py::module_local())
        .def("__call__", [](const PyOdeSolution<T, N>& self, const T& t){return self(t);})
        .def("__call__", [](const PyOdeSolution<T, N>& self, const py::array_t<T>& array){return self(array);});

    py::class_<PySolverState<T, N>>(m, "SolverState", py::module_local())
        .def_property_readonly("t", [](const PySolverState<T, N>& self){return self.t;})
        .def_property_readonly("q", [](const PySolverState<T, N>& self){return to_numpy<T>(self.q, self.shape);})
        .def_property_readonly("event", [](const PySolverState<T, N>& self){return self.event;})
        .def_property_readonly("diverges", [](const PySolverState<T, N>& self){return self.diverges;})
        .def_property_readonly("is_running", [](const PySolverState<T, N>& self){return self.is_running;})
        .def_property_readonly("is_dead", [](const PySolverState<T, N>& self){return self.is_dead;})
        .def_property_readonly("N", [](const PySolverState<T, N>& self){return self.Nt;})
        .def_property_readonly("message", [](const PySolverState<T, N>& self){return self.message;})
        .def("show", [](const PySolverState<T, N>& self){return self.show();});


    py::class_<PyODE<T, N>>(m, "LowLevelODE", py::module_local())
        .def(py::init<PyFuncWrapper<T, N>, T, py::object, T, T, T, T, T, py::tuple, PyEventWrapper<T, N>*, py::str>(),
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
            py::arg("events")=nullptr,
            py::arg("method")="RK45")
        .def(py::init<py::object, T, py::object, T, T, T, T, T, py::tuple, py::object, py::str>(),
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
            py::arg("events")=nullptr,
            py::arg("method")="RK45")
        .def("integrate", &PyODE<T, N>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("event_options")=py::dict(),
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("rich_integrate", &PyODE<T, N>::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::dict(),
            py::arg("max_prints")=0)
        .def("go_to", &PyODE<T, N>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("max_frames")=-1,
        py::arg("event_options")=py::dict(),
        py::arg("max_prints")=0,
        py::arg("include_first")=false)
        .def("copy", [](const PyODE<T, N>& self){return PyODE<T, N>(self);})
        .def("advance", [](PyODE<T, N>& self){return self.ode->advance();})
        .def("state", &PyODE<T, N>::py_state)
        .def("save_data", [](PyODE<T, N>& self, py::str save_dir){return self.ode->save_data(save_dir.cast<std::string>());}, py::arg("save_dir"))
        .def("clear", [](PyODE<T, N>& self){self.ode->clear();})
        .def("event_data", &PyODE<T, N>::event_data, py::arg("event"))
        .def_property_readonly("dim", [](const PyODE<T, N>& self){return self.ode->solver().Nsys();})
        .def_property_readonly("t", &PyODE<T, N>::t_array)
        .def_property_readonly("q", &PyODE<T, N>::q_array)
        .def_property_readonly("event_map", [](const PyODE<T, N>& self){return to_PyDict(self.ode->event_map());})
        .def_property_readonly("runtime", [](const PyODE<T, N>& self){return self.ode->runtime();})
        .def_property_readonly("diverges", [](const PyODE<T, N>& self){return self.ode->diverges();})
        .def_property_readonly("is_dead", [](const PyODE<T, N>& self){return self.ode->is_dead();});
    
    py::class_<PyVarODE<T, N>, PyODE<T, N>>(m, "VariationalLowLevelODE", py::module_local())
        .def(py::init<PyFuncWrapper<T, N>, T, py::object, T, T, T, T, T, T, py::tuple, PyEventWrapper<T, N>*, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("events")=nullptr,
            py::arg("method")="RK45")
        .def(py::init<py::object, T, py::object, T, T, T, T, T, T, py::tuple, py::object, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("events")=nullptr,
            py::arg("method")="RK45")
        .def_property_readonly("t_lyap", &PyVarODE<T, N>::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE<T, N>::py_lyap)
        .def("copy", [](const PyVarODE<T, N>& self){return PyVarODE<T, N>(self);});

    py::class_<PyEventWrapper<T, N>>(m, "LowLevelEventArray", py::module_local())
        .def(py::init<py::capsule, size_t, size_t>(), py::arg("pointer"), py::arg("q_size"), py::arg("args_size"));

    py::class_<PyFuncWrapper<T, N>>(m, "LowLevelFunction", py::module_local())
        .def(py::init<py::capsule, size_t, size_t>(), py::arg("pointer"), py::arg("q_size"), py::arg("args_size"))
        .def("__call__", [](const PyFuncWrapper<T, N>& self, const T& t, const py::object& q, py::args py_args){
            return self.call(t, q, toCPP_Array<T, std::vector<T>>(py_args));
        }, py::arg("t"), py::arg("q"));

    m.def("integrate_all", [](py::object list, const T& interval, const int& max_frames, const py::dict& event_options, const int& threads, const bool& display_progress){
            std::vector<ODE<T, N>*> array;
            for (const py::handle& item : list) {
                array.push_back(item.cast<PyODE<T, N>&>().ode);
            }
            integrate_all(array, interval, max_frames, to_Options(event_options), threads, display_progress);
        }, py::arg("ode_array"), py::arg("interval"), py::arg("max_frames")=-1, py::arg("event_options")=py::dict(), py::arg("threads")=-1, py::arg("display_progress")=false);
}

#endif
