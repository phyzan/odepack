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

py::dict to_PyDict(const std::map<std::string, std::vector<size_t>>& _map);

template<typename T, class Ty>
Ty toCPP_Array(const py::iterable& obj);

template<typename T, int N>
std::vector<T> flatten(const std::vector<vec<T, N>>&);

template<typename T, int N>
std::vector<std::unique_ptr<Event<T, N>>> to_Events(py::iterable events, const _Shape& shape, py::iterable args);

struct PyOptions : public EventOptions{

    PyOptions(std::string name, int max_events, bool terminate, int period) : EventOptions{name, max_events, terminate, period} {}

};

std::vector<EventOptions> to_Options(py::iterable d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        PyOptions opt = py::cast<PyOptions>(item);
        result.push_back(EventOptions(opt));
    }
    result.shrink_to_fit();
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

template<typename T>
py::tuple as_tuple(const T* x, int size){
    py::tuple tup(size);
    for (int i=0; i<size; i++){
        tup[i] = x[i];
    }
    return tup;
}

struct PyStruct{

    py::function rhs;
    py::function jac;
    py::function mask;
    py::function event;
    _Shape shape = {};
    py::tuple py_args = py::make_tuple();

};

template<typename T>
void arrcpy(T* dst, const T* src, size_t size){
    for (size_t i=0; i<size; i++){
        dst[i] = src[i];
    }
}

template<typename T>
void arrcpy(T* dst, const void* src, size_t size){
    const T* data = static_cast<const T*>(src);
    arrcpy(dst, data, size);
}

template<typename T>
void py_rhs(T* res, const T& t, const T* q, const T* args, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.rhs(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_jac(T* res, const T& t, const T* q, const T* args, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.jac(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
void py_mask(T* res, const T& t, const T* q, const T* args, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    py::array_t<T> pyres = p.mask(t, py::array_t<T>(p.shape, q), *p.py_args);
    arrcpy(res, pyres.data(), pyres.size());
}

template<typename T>
T py_event(const T& t, const T* q, const T* args, const void* obj){
    //args should always be the same as p.py_args
    const PyStruct& p = *reinterpret_cast<const PyStruct*>(obj);
    return p.event(t, py::array_t<T>(p.shape, q), *p.py_args).template cast<T>();
}


template<typename T, int N>
class PyEvent {

public:

    PyEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, T event_tol, size_t Nsys, size_t Nargs) : _name(name), _dir(sgn(dir)), _hide_mask(hide_mask), _event_tol(event_tol), _Nsys(Nsys), _Nargs(Nargs){
        if (py::isinstance<py::capsule>(when)){
            this->_when = open_capsule<ObjFun<T>>(when);
        }
        else if (py::isinstance<py::function>(when)){
            data.event = when;
            this->_when = py_event<T>;
        }

        if (py::isinstance<py::capsule>(mask)){
            this->_mask = open_capsule<Func<T>>(mask);
        }
        else if (py::isinstance<py::function>(mask)){
            data.mask = mask;
            this->_mask = py_mask<T>;
        }

    }

    DEFAULT_RULE_OF_FOUR(PyEvent);

    const std::string&  name() const{ return _name;}

    bool                hide_mask() const {return _hide_mask;}

    T                   event_tol() const {return _event_tol;}

    bool                is_lowlevel() const{
        return data.event.is_none() && data.mask.is_none();
    }

    void                check_sizes(size_t Nsys, size_t Nargs) const{
        std::vector<py::function> funcs({data.event, data.mask});
        for (const py::function& item : funcs){
            if (item.is_none()){
                //meaning that the function is lowlevel
                if (_Nsys != Nsys){
                    throw py::value_error("The event named \""+this->name()+"\" can only be applied on an ode of system size "+std::to_string(_Nsys)+", not "+std::to_string(Nsys));
                }
                else if (_Nargs != Nargs){
                    throw py::value_error("The event named \""+this->name()+"\" can only accept "+std::to_string(_Nargs)+" extra args, not "+std::to_string(Nargs));
                }
            }
        }
    }

    virtual std::unique_ptr<Event<T, N>> toEvent(const _Shape& shape, py::tuple args) {
        if (this->is_lowlevel()){
            for (const py::handle& arg : args){
                if (!PyNumber_Check(arg.ptr())){
                    throw py::value_error("All args must be numbers");
                }
            }
        }
        data.py_args = args;
        data.shape = shape;
        return std::make_unique<ObjectOwningEvent<PreciseEvent<T, N>, PyStruct>>(data, this->name(), this->_when, _dir, this->_mask, this->hide_mask(), this->event_tol());
    }

    virtual ~PyEvent(){}

protected:

    std::string _name;
    int _dir = 0;
    bool _hide_mask;
    T _event_tol;

    ObjFun<T> _when = nullptr;
    Func<T> _mask = nullptr;

    size_t _Nsys;
    size_t _Nargs;

    PyStruct data;
};


template<typename T, int N>
class PyPerEvent : public PyEvent<T, N>{

public:

    PyPerEvent(std::string name, T period, py::object start, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs):PyEvent<T, N>(name, py::function(), 0, mask, hide_mask, 0, Nsys, Nargs), _period(period), _start(start.is_none() ? inf<T>() : start.cast<T>()){}

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    std::unique_ptr<Event<T, N>> toEvent(const _Shape& shape, py::tuple args) override {

        if (this->is_lowlevel()){
            for (const py::handle& arg : args){
                if (!PyNumber_Check(arg.ptr())){
                    throw py::value_error("All args must be numbers");
                }
            }
        }
        this->data.py_args = args;
        this->data.shape = shape;
        return std::make_unique<ObjectOwningEvent<PeriodicEvent<T, N>, PyStruct>>(this->data, this->name(), _period, _start, this->_mask, this->hide_mask());
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
struct PyFuncWrapper{

    Func<T> rhs;
    size_t Nsys;
    std::vector<size_t> output_shape;
    size_t Nargs;
    size_t output_size;


    PyFuncWrapper(py::capsule obj, size_t Nsys, py::iterable output_shape, size_t Nargs) : rhs(open_capsule<Func<T>>(obj)), Nsys(Nsys), Nargs(Nargs) {
        this->output_shape = toCPP_Array<size_t, std::vector<size_t>>(output_shape);
        size_t s = 1;
        for (size_t i=0; i<this->output_shape.size(); i++){
            s *= this->output_shape[i];
        }
        this->output_size = s;
    }

    py::array_t<T> call(const T& t, const py::iterable& py_q, py::args py_args) const {
        std::vector<T> args =  toCPP_Array<T, std::vector<T>>(py_args);
        vec<T, N> q = toCPP_Array<T, vec<T, N>>(py_q);
        vec<T, N> res(output_size);
        if (static_cast<size_t>(q.size()) != Nsys || args.size() != Nargs){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        this->rhs(res.data(), t, q.data(), args.data(), nullptr);
        return to_numpy<T>(res, output_shape);
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

    PyODE(py::function f, T t0, py::iterable q0, py::object jacobian, T rtol, T atol, T min_step, T max_step, T first_step, py::iterable py_args, py::iterable events, py::str method){
        OdeData<T> ode_rhs = this->_init(f, q0, jacobian, py_args, events);
        std::vector<T> args = toCPP_Array<T, std::vector<T>>(py_args);
        std::unique_ptr<OdeSolver<T, N>> solver = get_solver<T, N>(method.cast<std::string>(), ode_rhs, t0, toCPP_Array<T, vec<T, N>>(q0), rtol, atol, min_step, max_step, first_step, args, to_Events<T, N>(events, shape(q0), py_args));
        ode = new ODE<T, N>(*solver);
    }

protected:
    PyODE(){}//derived classes manage ode and q0_shape creation

    OdeData<T> _init(py::function f, py::iterable q0, py::object jacobian, py::iterable py_args, py::iterable events){

        data.shape = shape(q0);
        data.py_args = py::tuple(py_args);
        size_t _size = prod(data.shape);
        std::vector<T> args = toCPP_Array<T, std::vector<T>>(py_args);
        OdeData<T> ode_rhs = {nullptr, nullptr, &data};
        if (py::isinstance<PyFuncWrapper<T, N>>(f)){
            PyFuncWrapper<T, N>& _f = f.cast<PyFuncWrapper<T, N>&>();
            ode_rhs.rhs = _f.rhs;
            if (_f.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size");
            }
            else if (_f.Nargs != args.size()){
                throw py::value_error("The array size of the given extra args differs from the number of args specified for this ode system");
            }
        }
        else{
            data.rhs = f;
            ode_rhs.rhs = py_rhs;
        }
        
        if (py::isinstance<PyFuncWrapper<T, N>>(jacobian)){
            PyFuncWrapper<T, N>& _j = jacobian.cast<PyFuncWrapper<T, N>&>();
            ode_rhs.jacobian = _j.rhs;
            if (_j.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size that applied in the provided jacobian");
            }
            else if (_j.Nargs != args.size()){
                throw py::value_error("The array size of the given extra args differs from the number of args specified for the provided jacobian");
            }
        }
        else if (!jacobian.is_none()){
            data.jac = jacobian;
            ode_rhs.jacobian = py_jac;
        }

        for (py::handle ev : events){
            if (!py::isinstance<PyEvent<T, N>>(ev)){
                throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class");
            }
            const PyEvent<T, N>& _ev = ev.cast<const PyEvent<T, N>&>();
            _ev.check_sizes(_size, args.size());

        }
        return ode_rhs;

    }

public:

    PyODE(const PyODE<T, N>& other) : ode(other.ode->clone()), data(other.data){
        ode->set_obj(&data);
    }

    PyODE(PyODE<T, N>&& other):ode(other.ode), data(other.data){
        other.ode = nullptr;
        ode->set_obj(&data);
    }

    PyODE<T, N>& operator=(const PyODE<T, N>& other){
        if (&other == this){
            return *this;
        }
        delete ode;
        ode = other.ode->clone();
        data = other.data;
        ode->set_obj(&data);
        return *this;
    }

    PyODE<T, N>& operator=(PyODE<T, N>&& other){
        if (&other == this){
            return *this;
        }
        ode = other.ode;
        other.ode = nullptr;
        data = other.data;
        ode->set_obj(&data);
        return *this;
    }

    virtual ~PyODE(){
        delete ode;
    }

    PySolverState<T, N> py_state() const{
        SolverState<T, N> s = ode->state();
        return PySolverState<T, N>(s.t, s.q, s.habs, s.event, s.diverges, s.is_running, s.is_dead, s.Nt, s.message, data.shape); 
    }

    PyOdeResult<T, N> py_integrate(const T& interval, const int max_frames, py::iterable event_options, const int& max_prints, const bool& include_first){
        OdeResult<T, N> res = ode->integrate(interval, max_frames, to_Options(event_options), max_prints, include_first);
        PyOdeResult<T, N> py_res(res, data.shape);
        return py_res;
    }

    PyOdeSolution<T, N> py_rich_integrate(const T& interval, py::iterable event_options, const int& max_prints){
        OdeSolution<T, N> res = ode->rich_integrate(interval, to_Options(event_options), max_prints);
        PyOdeSolution<T, N> py_res(res, data.shape);
        return py_res;
    }


    PyOdeResult<T, N> py_go_to(const T& t, const int max_frames, py::iterable event_options, const int& max_prints, const bool& include_first){
        OdeResult<T, N> res = ode->go_to(t, max_frames, to_Options(event_options), max_prints, include_first);
        PyOdeResult<T, N> py_res(res, data.shape);
        return py_res;
    }

    py::array_t<T> t_array()const{
        py::array_t<T> res = to_numpy<T>(ode->t());
        return res;
    }

    py::array_t<T> q_array()const{
        py::array_t<T> res = to_numpy<T>(flatten<T, N>(ode->q()), getShape(ode->t().size(), data.shape));
        return res;
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = ode->t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        std::vector<vec<T, N>> q_data = ode->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(flatten<T, N>(q_data), getShape(q_data.size(), data.shape));
        return py::make_tuple(t_res, q_res);
    }
    ODE<T, N>* ode;
    PyStruct data;
};


template<typename T, int N>
class PyVarODE : public PyODE<T, N>{

public:

    PyVarODE(py::function f, T t0, py::iterable q0, T period, py::object jac, T rtol, T atol, T min_step, T max_step, T first_step, py::iterable args, py::iterable events, py::str method):PyODE<T, N>(){
        OdeData<T> ode_rhs = this->_init(f, q0, jac, args, events);
        vec<T, N> q0_ = toCPP_Array<T, vec<T, N>>(q0);
        std::vector<T> args_ = toCPP_Array<T, std::vector<T>>(args);
        this->ode = new VariationalODE<T, N>(ode_rhs, t0, q0_, period, rtol, atol, min_step, max_step, first_step, args_, to_Events<T, N>(events, shape(q0), args), method.cast<std::string>());
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


template<typename T, class Ty>
Ty toCPP_Array(const py::iterable& obj) {
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

template<typename T, int N>
std::vector<std::unique_ptr<Event<T, N>>> to_Events(py::iterable events, const _Shape& shape, py::iterable args){   
    if (events.is_none()){
        return {};
    }
    std::vector<std::unique_ptr<Event<T, N>>> res;
    for (py::handle item : events){
        res.push_back(item.cast<PyEvent<T, N>&>().toEvent(shape, args));
    }
    return res;
}

template<typename T, int N>
void define_ode_module(py::module& m) {

    py::class_<PyOptions>(m, "EventOpt")
        .def(py::init<py::str, int, bool, int>(), py::arg("name"), py::arg("max_events")=-1, py::arg("terminate")=false, py::arg("period")=1);

    py::class_<PyEvent<T, N>>(m, "Event")
        .def(py::init<std::string, py::object, int, py::object, bool, T, size_t, size_t>(),
        py::arg("name"),
        py::arg("when"),
        py::arg("dir")=0,
        py::arg("mask")=py::none(),
        py::arg("hide_mask")=false,
        py::arg("event_tol")=1e-12,
        py::arg("__Nsys")=0,
        py::arg("__Nargs")=0)
        .def_property_readonly("name", [](const PyEvent<T, N>& self){
            return self.name();
        })
        .def_property_readonly("hide_mask", [](const PyEvent<T, N>& self){
            return self.hide_mask();
        });
        
    py::class_<PyPerEvent<T, N>, PyEvent<T, N>>(m, "PeriodicEvent")
        .def(py::init<std::string, T, py::object, py::object, bool, size_t, size_t>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("start")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("period", [](const PyPerEvent<T, N>& self){
                return self.period();})
        .def_property_readonly("start", [](const PyPerEvent<T, N>& self){
            return self.start();});

    py::class_<PyOdeResult<T, N>>(m, "OdeResult")
        .def(py::init<PyOdeResult<T, N>>(), py::arg("result"))
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
    
    py::class_<PyOdeSolution<T, N>, PyOdeResult<T, N>>(m, "OdeSolution")
        .def(py::init<PyOdeSolution<T, N>>(), py::arg("result"))
        .def("__call__", [](const PyOdeSolution<T, N>& self, const T& t){return self(t);})
        .def("__call__", [](const PyOdeSolution<T, N>& self, const py::array_t<T>& array){return self(array);});

    py::class_<PySolverState<T, N>>(m, "SolverState")
        .def_property_readonly("t", [](const PySolverState<T, N>& self){return self.t;})
        .def_property_readonly("q", [](const PySolverState<T, N>& self){return to_numpy<T>(self.q, self.shape);})
        .def_property_readonly("event", [](const PySolverState<T, N>& self){return self.event;})
        .def_property_readonly("diverges", [](const PySolverState<T, N>& self){return self.diverges;})
        .def_property_readonly("is_running", [](const PySolverState<T, N>& self){return self.is_running;})
        .def_property_readonly("is_dead", [](const PySolverState<T, N>& self){return self.is_dead;})
        .def_property_readonly("N", [](const PySolverState<T, N>& self){return self.Nt;})
        .def_property_readonly("message", [](const PySolverState<T, N>& self){return self.message;})
        .def("show", [](const PySolverState<T, N>& self){return self.show();});


    py::class_<PyODE<T, N>>(m, "LowLevelODE")
        .def(py::init<py::function, T, py::iterable, py::object, T, T, T, T, T, py::iterable, py::iterable, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45")
        .def(py::init<PyODE<T, N>>(), py::arg("ode"))
        .def("integrate", &PyODE<T, N>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("max_frames")=-1,
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0,
            py::arg("include_first")=false)
        .def("rich_integrate", &PyODE<T, N>::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("go_to", &PyODE<T, N>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("max_frames")=-1,
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0,
        py::arg("include_first")=false)
        .def("copy", [](const PyODE<T, N>& self){return PyODE<T, N>(self);})
        .def("reset", [](PyODE<T, N>& self){self.ode->reset();})
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
    
    py::class_<PyVarODE<T, N>, PyODE<T, N>>(m, "VariationalLowLevelODE")
        .def(py::init<py::function, T, py::iterable, T, py::object, T, T, T, T, T, py::iterable, py::iterable, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-6,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45")
        .def(py::init<PyVarODE<T, N>>(), py::arg("ode"))
        .def_property_readonly("t_lyap", &PyVarODE<T, N>::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE<T, N>::py_lyap)
        .def("copy", [](const PyVarODE<T, N>& self){return PyVarODE<T, N>(self);});

    py::class_<PyFuncWrapper<T, N>>(m, "LowLevelFunction")
        .def(py::init<py::capsule, size_t, py::iterable, size_t>(), py::arg("pointer"), py::arg("input_size"), py::arg("output_shape"), py::arg("Nargs"))
        .def("__call__", &PyFuncWrapper<T, N>::call, py::arg("t"), py::arg("q"));

    m.def("integrate_all", [](py::object list, const T& interval, const int& max_frames, py::iterable event_options, const int& threads, const bool& display_progress){
            std::vector<ODE<T, N>*> array;
            for (const py::handle& item : list) {
                array.push_back(item.cast<PyODE<T, N>&>().ode);
            }
            integrate_all(array, interval, max_frames, to_Options(event_options), threads, display_progress);
        }, py::arg("ode_array"), py::arg("interval"), py::arg("max_frames")=-1, py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);
}

#endif
