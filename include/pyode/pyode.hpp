#ifndef PYODE_HPP
#define PYODE_HPP

#include "pytools.hpp"

#define THIS_SOLUTION static_cast<const OdeSolution<T, 0>*>(this->res)


/*
FORWARD DECLARATIONS & INTERFACES
*/

template<typename T>
struct PyFuncWrapper{

    Func<T> rhs;
    size_t Nsys;
    std::vector<py::ssize_t> output_shape;
    size_t Nargs;
    size_t output_size;

    PyFuncWrapper(py::capsule obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs);

    py::object call(const T& t, const py::iterable& py_q, py::args py_args) const;
};


template<typename T>
class PyEvent{

public:

    PyEvent(std::string name, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs);

    py::str             name() const;

    py::bool_           hide_mask() const;

    py::bool_           is_lowlevel() const;

    void                check_sizes(size_t Nsys, size_t Nargs) const;

    virtual std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) = 0;

    virtual ~PyEvent(){}

protected:

    std::string _name;
    bool _hide_mask;
    Func<T> _mask = nullptr;
    size_t _Nsys;
    size_t _Nargs;
    PyStruct data;
};


template<typename T>
class PyPrecEvent : public PyEvent<T> {

public:

    PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, T event_tol, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPrecEvent);

    py::object event_tol() const;

    std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

protected:

    int _dir = 0;
    T _event_tol;
    ObjFun<T> _when = nullptr;
};


template<typename T>
class PyPerEvent : public PyEvent<T>{

public:

    PyPerEvent(std::string name, T period, py::object start, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

    py::object period() const;

    py::object start() const;

private:
    T _period;
    T _start;

};

template<typename T>
struct PySolver {

    PySolver(const py::object& f, const py::object& jac, T t0, py::iterable py_q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name);

    PySolver(const OdeSolver<T, 0>* other, PyStruct py_data) : s(other->clone()), data(std::move(py_data)){
        s->set_obj(&data);
    }

    PySolver(const PySolver& other);

    PySolver(PySolver&& other) noexcept;

    PySolver& operator=(const PySolver& other);

    PySolver& operator=(PySolver&& other) noexcept;

    OdeSolver<T, 0>* operator->();

    const OdeSolver<T, 0>* operator->() const;

    virtual ~PySolver();

    OdeSolver<T, 0>* s = nullptr;
    PyStruct data;
};

template<typename T>
struct PyRK23 : public PySolver<T>{

    PyRK23(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events);

    PyRK23(const OdeSolver<T, 0>* other, PyStruct py_data) : PySolver<T>(other, py_data) {}
};


template<typename T>
struct PyRK45 : public PySolver<T>{

    PyRK45(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events);

    PyRK45(const OdeSolver<T, 0>* other, PyStruct py_data) : PySolver<T>(other, py_data) {}
};

template<typename T>
struct PyDOP853 : public PySolver<T>{

    PyDOP853(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events);

    PyDOP853(const OdeSolver<T, 0>* other, PyStruct py_data) : PySolver<T>(other, py_data) {}
};


template<typename T>
struct PyBDF : public PySolver<T>{

    PyBDF(const py::object& f, const py::object& jac, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events);

    PyBDF(const OdeSolver<T, 0>* other, PyStruct py_data) : PySolver<T>(other, py_data) {}
};


template<typename T>
struct PyOdeResult{

    PyOdeResult(const OdeResult<T, 0>& r, const std::vector<py::ssize_t>& q0_shape);

    PyOdeResult(const PyOdeResult& other);

    PyOdeResult(PyOdeResult&& other) noexcept;

    virtual ~PyOdeResult();

    PyOdeResult& operator=(const PyOdeResult& other);

    PyOdeResult& operator=(PyOdeResult&& other) noexcept;

    py::tuple event_data(const py::str& event) const;

    OdeResult<T, 0>* res;
    std::vector<py::ssize_t> q0_shape;

};

template<typename T>
struct PyOdeSolution : public PyOdeResult<T>{

    PyOdeSolution(const OdeSolution<T, 0>& res, const std::vector<py::ssize_t>& q0_shape);

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    py::object operator()(const T& t) const;

    py::object operator()(const py::array& py_array) const;

    size_t nsys;

};



template<typename T>
class PyODE{

public:

    PyODE(py::object f, T t0, py::iterable py_q0, py::object jacobian, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method);

protected:
    PyODE() = default;//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE<T>& other);

    PyODE(PyODE<T>&& other) noexcept;

    PyODE<T>& operator=(const PyODE<T>& other);

    PyODE<T>& operator=(PyODE<T>&& other) noexcept;

    virtual ~PyODE();

    py::object py_integrate(const T& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object py_rich_integrate(const T& interval, const py::iterable& event_options, int max_prints);

    py::object py_go_to(const T& t, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object t_array() const;

    py::object q_array() const;

    py::tuple event_data(const py::str& event) const;

    py::object solver_copy() const;

    ODE<T, 0>* ode = nullptr;
    PyStruct data;
};


template<typename T>
class PyVarODE : public PyODE<T>{

public:

    PyVarODE(py::object f, T t0, py::iterable q0, T period, py::object jac, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method);

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    VariationalODE<T, 0>& varode();

    const VariationalODE<T, 0>& varode() const;

    py::object py_t_lyap() const;

    py::object py_lyap() const;

    py::object py_kicks() const;

};

inline void define_event_opt(py::module& m);

template<typename T>
void define_ode_module(py::module& m, const std::string& suffix = "");


/*
IMPLEMENTATIONS
*/

// PyFuncWrapper::PyFuncWrapper
template<typename T>
PyFuncWrapper<T>::PyFuncWrapper(py::capsule obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs) : rhs(open_capsule<Func<T>>(obj)), Nsys(static_cast<size_t>(Nsys)), output_shape(static_cast<size_t>(output_shape.size())), Nargs(static_cast<size_t>(Nargs)) {
    copy_array(this->output_shape.data(), output_shape.data(), this->output_shape.size());
    long s = 1;
    for (long i : this->output_shape){
        s *= i;
    }
    this->output_size = s;
}

// PyFuncWrapper::call
template<typename T>
py::object PyFuncWrapper<T>::call(const T& t, const py::iterable& py_q, py::args py_args) const {
    auto q= toCPP_Array<T, Array1D<T>>(py_q);
    if (static_cast<size_t>(q.size()) != Nsys || py_args.size() != Nargs){
        throw py::value_error("Invalid array sizes in ode function call");
    }
    auto args = toCPP_Array<T, std::vector<T>>(py_args);
    Array<T> res(output_shape);
    this->rhs(res.data(), t, q.data(), args.data(), nullptr);
    return py::cast(res);
}

// PyEvent::PyEvent
template<typename T>
PyEvent<T>::PyEvent(std::string name, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs) : _name(std::move(name)), _hide_mask(hide_mask), _Nsys(Nsys), _Nargs(Nargs){
    if (py::isinstance<py::capsule>(mask)){
        this->_mask = open_capsule<Func<T>>(mask);
    }
    else if (py::isinstance<py::function>(mask)){
        data.mask = mask;
        this->_mask = py_mask<T>;
    }
}

// PyEvent::name
template<typename T>
py::str PyEvent<T>::name() const{
    return _name;
}

// PyEvent::hide_mask
template<typename T>
py::bool_ PyEvent<T>::hide_mask() const {
    return _hide_mask;
}

// PyEvent::is_lowlevel
template<typename T>
py::bool_ PyEvent<T>::is_lowlevel() const{
    return data.event.is_none() && data.mask.is_none();
}

// PyEvent::check_sizes
template<typename T>
void PyEvent<T>::check_sizes(size_t Nsys, size_t Nargs) const{
    std::vector<py::function> funcs({data.event, data.mask});
    for (const py::function& item : funcs){
        if (item.is_none()){
            //meaning that the function is lowlevel
            if (_Nsys != Nsys){
                throw py::value_error("The event named \""+this->_name+"\" can only be applied on an ode of system size "+std::to_string(_Nsys)+", not "+std::to_string(Nsys));
            }
            else if (_Nargs != Nargs){
                throw py::value_error("The event named \""+this->_name+"\" can only accept "+std::to_string(_Nargs)+" extra args, not "+std::to_string(Nargs));
            }
        }
    }
}

// PyPrecEvent::PyPrecEvent
template<typename T>
PyPrecEvent<T>::PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, T event_tol, size_t Nsys, size_t Nargs) : PyEvent<T>(name, mask, hide_mask, Nsys, Nargs), _dir(sgn(dir)), _event_tol(event_tol){
    if (py::isinstance<py::capsule>(when)){
        this->_when = open_capsule<ObjFun<T>>(when);
    }
    else if (py::isinstance<py::function>(when)){
        this->data.event = when;
        this->_when = py_event<T>;
    }

}

// PyPrecEvent::event_tol
template<typename T>
py::object PyPrecEvent<T>::event_tol() const {
    return py::cast(_event_tol);
}

// PyPrecEvent::toEvent
template<typename T>
std::unique_ptr<Event<T, 0>> PyPrecEvent<T>::toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) {
    if (this->is_lowlevel()){
        for (const py::handle& arg : args){
            if (PyNumber_Check(arg.ptr()) == 0){
                throw py::value_error("All args must be numbers");
            }
        }
    }
    this->data.py_args = args;
    this->data.shape = shape;
    return std::make_unique<ObjectOwningEvent<PreciseEvent<T, 0>, PyStruct>>(this->data, this->name(), this->_when, _dir, this->_mask, this->hide_mask(), this->_event_tol);
}

// PyPerEvent::PyPerEvent
template<typename T>
PyPerEvent<T>::PyPerEvent(std::string name, T period, py::object start, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs):PyEvent<T>(name, mask, hide_mask, Nsys, Nargs), _period(period), _start(start.is_none() ? inf<T>() : start.cast<T>()){}

// PyPerEvent::toEvent
template<typename T>
std::unique_ptr<Event<T, 0>> PyPerEvent<T>::toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) {

    if (this->is_lowlevel()){
        for (const py::handle& arg : args){
            if (PyNumber_Check(arg.ptr())==0){
                throw py::value_error("All args must be numbers");
            }
        }
    }
    this->data.py_args = args;
    this->data.shape = shape;
    return std::make_unique<ObjectOwningEvent<PeriodicEvent<T, 0>, PyStruct>>(this->data, this->name(), _period, _start, this->_mask, this->hide_mask());
}

// PyPerEvent::period
template<typename T>
py::object PyPerEvent<T>::period() const{
    return py::cast(_period);
}

// PyPerEvent::start
template<typename T>
py::object PyPerEvent<T>::start() const{
    return py::cast(_start);
}

// to_Options
inline std::vector<EventOptions> to_Options(const py::iterable& d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        auto opt = py::cast<EventOptions>(item);
        result.emplace_back(opt);
    }
    result.shrink_to_fit();
    return result;
}

// to_Events
template<typename T>
std::vector<std::unique_ptr<Event<T, 0>>> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, py::iterable args){
    if (events.is_none()){
        return {};
    }
    std::vector<std::unique_ptr<Event<T, 0>>> res;
    for (py::handle item : events){
        res.push_back(item.cast<PyEvent<T>&>().toEvent(shape, args));
    }
    return res;
}

// init_ode_data
template<typename T>
OdeData<T> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    std::string scalar_type;
    if constexpr (std::is_same_v<T, double>){
        scalar_type = "double";
    }
    else if constexpr (std::is_same_v<T, long double>){
        scalar_type = "long double";
    }
    else if constexpr (std::is_same_v<T, float>){
        scalar_type = "float";
    }
    else if constexpr (std::is_same_v<T, mpfr::mpreal>){
        scalar_type = "mpreal";
    }
    else{
        static_assert(false, "Unsupported scalar type T");
    }
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);

    bool f_is_compiled = py::isinstance<PyFuncWrapper<T>>(f) || py::isinstance<py::capsule>(f);
    bool jac_is_compiled = !jacobian.is_none() && (py::isinstance<PyFuncWrapper<T>>(jacobian) || py::isinstance<py::capsule>(jacobian));
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<T> ode_rhs = {nullptr, nullptr, &data};
    if (f_is_compiled){
        if (py::isinstance<PyFuncWrapper<T>>(f)){
            //safe approach
            PyFuncWrapper<T>& _f = f.cast<PyFuncWrapper<T>&>();
            ode_rhs.rhs = _f.rhs;
            if (_f.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size");
            }
            else if (_f.Nargs != args.size()){
                throw py::value_error("The number of the provided extra args (" + std::to_string(args.size()) + ") differs from the number of args specified for this ode system ("+std::to_string(_f.Nargs)+").");
            }
        }
        else{
            ode_rhs.rhs = open_capsule<Func<T>>(f.cast<py::capsule>());
        }
    }
    else{
        data.rhs = f;
        ode_rhs.rhs = py_rhs;
    }
    if (jac_is_compiled){
        if (py::isinstance<PyFuncWrapper<T>>(jacobian)){
            //safe approach
            PyFuncWrapper<T>& _j = jacobian.cast<PyFuncWrapper<T>&>();
            ode_rhs.jacobian = _j.rhs;
            if (_j.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size that applied in the provided jacobian");
            }
            else if (_j.Nargs != args.size()){
                throw py::value_error("The array size of the given extra args differs from the number of args specified for the provided jacobian");
            }
        }
        else{
            ode_rhs.jacobian = open_capsule<Func<T>>(jacobian.cast<py::capsule>());
        }
    }
    else if (!jacobian.is_none()){
        data.jac = jacobian;
        ode_rhs.jacobian = py_jac;
    }
    for (py::handle ev : events){
        if (!py::isinstance<PyEvent<T>>(ev)) {
            throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class with scalar type " + scalar_type + ", not " + py::str(ev.get_type()).cast<std::string>());
        }
        const PyEvent<T>& _ev = ev.cast<const PyEvent<T>&>();
        _ev.check_sizes(_size, args.size());

    }
    return ode_rhs;
}

// PySolver::PySolver (main constructor)
template<typename T>
PySolver<T>::PySolver(const py::object& f, const py::object& jac, T t0, py::iterable py_q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
    std::vector<T> args;
    OdeData<T> ode_data = init_ode_data<T>(this->data, args, f, py_q0, jac, py_args, py_events);
    auto safe_events = to_Events<T>(py_events, this->data.shape, py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i].get();
    }
    auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
    this->s = get_solver(name, ode_data, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, evs).release();
}

// PySolver::operator->
template<typename T>
OdeSolver<T, 0>* PySolver<T>::operator->(){
    return s;
}

// PySolver::operator-> const
template<typename T>
const OdeSolver<T, 0>* PySolver<T>::operator->() const{
    return s;
}

// PySolver::PySolver (copy constructor)
template<typename T>
PySolver<T>::PySolver(const PySolver& other) : s(other.s->clone()), data(other.data) {
    s->set_obj(&data);
}

// PySolver::PySolver (move constructor)
template<typename T>
PySolver<T>::PySolver(PySolver&& other) noexcept : s(other.s), data(std::move(other.data)) {
    other.s = nullptr;
    s->set_obj(&data);
}

// PySolver::operator= (copy assignment)
template<typename T>
PySolver<T>& PySolver<T>::operator=(const PySolver& other){
    if (&other != this){
        delete s;
        s = other.s->clone();
        data = other.data;
        s->set_obj(&data);
    }
    return *this;
}

// PySolver::operator= (move assignment)
template<typename T>
PySolver<T>& PySolver<T>::operator=(PySolver&& other) noexcept{
    if (&other != this){
        delete s;
        s = other.s;
        other.s = nullptr;
        data = std::move(other.data);
        s->set_obj(&data);
    }
    return *this;
}

// PySolver::~PySolver
template<typename T>
PySolver<T>::~PySolver(){
    delete s;
}

// PyRK23::PyRK23
template<typename T>
PyRK23<T>::PyRK23(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK23"){}

// PyRK45::PyRK45
template<typename T>
PyRK45<T>::PyRK45(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK45"){}

// PyDOP853::PyDOP853
template<typename T>
PyDOP853<T>::PyDOP853(const py::object& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}

// PyBDF::PyBDF
template<typename T>
PyBDF<T>::PyBDF(const py::object& f, const py::object& jac, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(f, jac, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}

// PyOdeResult::PyOdeResult (constructor from OdeResult)
template<typename T>
PyOdeResult<T>::PyOdeResult(const OdeResult<T, 0>& r, const std::vector<py::ssize_t>& q0_shape): res(r.clone()), q0_shape(q0_shape){}

// PyOdeResult::PyOdeResult (copy constructor)
template<typename T>
PyOdeResult<T>::PyOdeResult(const PyOdeResult& other) : res(other.res->clone()), q0_shape(other.q0_shape) {}

// PyOdeResult::PyOdeResult (move constructor)
template<typename T>
PyOdeResult<T>::PyOdeResult(PyOdeResult&& other) noexcept : res(other.res), q0_shape(std::move(other.q0_shape)) {
    other.res = nullptr;
}

// PyOdeResult::~PyOdeResult
template<typename T>
PyOdeResult<T>::~PyOdeResult(){
    delete res;
    res = nullptr;
}

// PyOdeResult::operator= (copy assignment)
template<typename T>
PyOdeResult<T>& PyOdeResult<T>::operator=(const PyOdeResult& other){
    if (&other != this){
        delete res;
        res = other.res->clone();
        q0_shape = other.q0_shape;
    }
    return *this;
}

// PyOdeResult::operator= (move assignment)
template<typename T>
PyOdeResult<T>& PyOdeResult<T>::operator=(PyOdeResult&& other) noexcept{
    if (&other != this){
        delete res;
        res = other.res;
        q0_shape = other.q0_shape;
        other.res = nullptr;
    }
    return *this;
}

// PyOdeResult::event_data
template<typename T>
py::tuple PyOdeResult<T>::event_data(const py::str& event) const{
    std::vector<T> t_data = this->res->t_filtered(event.cast<std::string>());
    Array<T> t_res(t_data.data());
    Array2D<T> q_data = this->res->q_filtered(event.cast<std::string>());
    return py::make_tuple(t_res, q_data);
}

// PyOdeSolution::PyOdeSolution
template<typename T>
PyOdeSolution<T>::PyOdeSolution(const OdeSolution<T, 0>& res, const std::vector<py::ssize_t>& q0_shape) : PyOdeResult<T>(res, q0_shape), nsys(prod(q0_shape)) {}

// PyOdeSolution::operator() (T)
template<typename T>
py::object PyOdeSolution<T>::operator()(const T& t) const{
    auto res = THIS_SOLUTION->operator()(t);
    return py::cast(Array<T>(res.data(), this->q0_shape));
}

// PyOdeSolution::operator() (py::array)
template<typename T>
py::object PyOdeSolution<T>::operator()(const py::array& py_array) const{
    const auto nt = size_t(py_array.size());
    std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
    final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
    Array<T> res(final_shape);
    const T* data = static_cast<const T*>(py_array.data());
    for (size_t i=0; i<nt; i++){
        copy_array(res.data()+i*nsys, THIS_SOLUTION->operator()(data[i]).data(), nsys);
    }
    return py::cast(res);
}

// PyODE::PyODE (main constructor)
template<typename T>
PyODE<T>::PyODE(py::object f, T t0, py::iterable py_q0, py::object jacobian, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method){

    std::vector<T> args;
    OdeData<T> ode_rhs = init_ode_data<T>(data,args, f, py_q0, jacobian, py_args, events);
    auto safe_events = to_Events<T>(events, shape(py_q0), py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i].get();
    }
    auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
    ode = new ODE<T, 0>(ode_rhs, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, evs, method);
}

// PyODE::PyODE (copy constructor)
template<typename T>
PyODE<T>::PyODE(const PyODE<T>& other) : ode(other.ode->clone()), data(other.data){
    ode->set_obj(&data);
}

// PyODE::PyODE (move constructor)
template<typename T>
PyODE<T>::PyODE(PyODE<T>&& other) noexcept :ode(other.ode), data(std::move(other.data)){
    other.ode = nullptr;
    ode->set_obj(&data);
}

// PyODE::operator= (copy assignment)
template<typename T>
PyODE<T>& PyODE<T>::operator=(const PyODE<T>& other){
    if (&other == this){
        return *this;
    }
    delete ode;
    ode = other.ode->clone();
    data = other.data;
    ode->set_obj(&data);
    return *this;
}

// PyODE::operator= (move assignment)
template<typename T>
PyODE<T>& PyODE<T>::operator=(PyODE<T>&& other) noexcept {
    if (&other == this){
        return *this;
    }
    delete ode;
    ode = other.ode;
    other.ode = nullptr;
    data = other.data;
    ode->set_obj(&data);
    return *this;
}

// PyODE::~PyODE
template<typename T>
PyODE<T>::~PyODE(){
    delete ode;
}

// PyODE::py_integrate
template<typename T>
py::object PyODE<T>::py_integrate(const T& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    OdeResult<T, 0> res = ode->integrate(interval, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
    PyOdeResult<T> py_res(res, data.shape);
    return py::cast(py_res);
}

// PyODE::py_rich_integrate
template<typename T>
py::object PyODE<T>::py_rich_integrate(const T& interval, const py::iterable& event_options, int max_prints){
    OdeSolution<T, 0> res = ode->rich_integrate(interval, to_Options(event_options), max_prints);
    PyOdeSolution<T> py_res(res, data.shape);
    return py::cast(py_res);
}

// PyODE::py_go_to
template<typename T>
py::object PyODE<T>::py_go_to(const T& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    OdeResult<T, 0> res = ode->go_to(t, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
    PyOdeResult<T> py_res(res, data.shape);
    return py::cast(py_res);
}

// PyODE::t_array
template<typename T>
py::object PyODE<T>::t_array() const{
    auto res = NdView<const T>(ode->t().data(), ode->t().size());
    return py::cast(res);
}

// PyODE::q_array
template<typename T>
py::object PyODE<T>::q_array() const{
    NdView<const T, Layout::C, 0, 0> res = ode->q();
    return py::cast(res);
}

// PyODE::event_data
template<typename T>
py::tuple PyODE<T>::event_data(const py::str& event) const{
    std::vector<T> t_data = ode->t_filtered(event.cast<std::string>());
    Array2D<T, 0, 0> q_data = ode->q_filtered(event.cast<std::string>());
    Array<T> q_res(q_data.data(), getShape(q_data.shape(0), data.shape), true);
    return py::make_tuple(t_data, q_res);
}

// PyODE::solver_copy
template<typename T>
py::object PyODE<T>::solver_copy() const{
    if (ode->solver()->name() == "RK45"){
        return py::cast(PyRK45<T>(ode->solver(), data));
    }
    else if (ode->solver()->name() == "DOP853"){
        return py::cast(PyDOP853<T>(ode->solver(), data));
    }
    else if (ode->solver()->name() == "RK23"){
        return py::cast(PyRK23<T>(ode->solver(), data));
    }
    else if (ode->solver()->name() == "BDF"){
        return py::cast(PyBDF<T>(ode->solver(), data));
    }
    else{
        throw py::value_error("Unregistered solver!");
    }
}

// PyVarODE::PyVarODE
template<typename T>
PyVarODE<T>::PyVarODE(py::object f, T t0, py::iterable q0, T period, py::object jac, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method):PyODE<T>(){
    std::vector<T> args;
    OdeData<T> ode_rhs = init_ode_data<T>(this->data, args, f, q0, jac, py_args, events);
    Array1D<T> q0_ = toCPP_Array<T, Array1D<T>>(q0);
    auto safe_events = to_Events<T>(events, shape(q0), py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i].get();
    }
    this->ode = new VariationalODE<T, 0>(ode_rhs, t0, q0_, period, rtol, atol, min_step, max_step, first_step, dir, args, evs, method.cast<std::string>());
}

// PyVarODE::varode
template<typename T>
VariationalODE<T, 0>& PyVarODE<T>::varode(){
    return *static_cast<VariationalODE<T, 0>*>(this->ode);
}

// PyVarODE::varode const
template<typename T>
const VariationalODE<T, 0>& PyVarODE<T>::varode() const {
    return *static_cast<VariationalODE<T, 0>*>(this->ode);
}

// PyVarODE::py_t_lyap
template<typename T>
py::object PyVarODE<T>::py_t_lyap() const{
    NdView<const T> res(varode().t_lyap().data(), varode().t_lyap().size());
    return py::cast(res);
}

// PyVarODE::py_lyap
template<typename T>
py::object PyVarODE<T>::py_lyap() const{
    NdView<const T> res(varode().lyap().data(), varode().t_lyap().size());
    return py::cast(res);
}

// PyVarODE::py_kicks
template<typename T>
py::object PyVarODE<T>::py_kicks() const{
    NdView<const T> res(varode().kicks().data(), varode().t_lyap().size());
    return py::cast(res);
}

// define_event_opt
inline void define_event_opt(py::module& m){
    py::class_<EventOptions>(m, "EventOpt")
        .def(py::init<const std::string&, int, bool, int>(),
             py::arg("name"),
             py::arg("max_events") = -1,
             py::arg("terminate") = false,
             py::arg("period") = 1)
        .def_readwrite("name", &EventOptions::name)
        .def_readwrite("max_events", &EventOptions::max_events)
        .def_readwrite("terminate", &EventOptions::terminate)
        .def_readwrite("period", &EventOptions::period);
}

// define_ode_module
template<typename T>
void define_ode_module(py::module& m, const std::string& suffix){

    py::class_<PyEvent<T>>(m, ("Event" + suffix).c_str())
        .def_property_readonly("name", [](const PyEvent<T>& self){
            return self.name();
        })
        .def_property_readonly("hides_mask", [](const PyEvent<T>& self){
            return self.hide_mask();
        });

    py::class_<PyPrecEvent<T>, PyEvent<T>>(m, ("PreciseEvent" + suffix).c_str())
        .def(py::init<std::string, py::object, int, py::object, bool, T, size_t, size_t>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("dir")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12,
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("event_tol", [](const PyPrecEvent<T>& self){
            return self.event_tol();
        });

    py::class_<PyPerEvent<T>, PyEvent<T>>(m, ("PeriodicEvent" + suffix).c_str())
        .def(py::init<std::string, T, py::object, py::object, bool, size_t, size_t>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("start")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("period", [](const PyPerEvent<T>& self){
                return self.period();})
        .def_property_readonly("start", [](const PyPerEvent<T>& self){
            return self.start();});

    py::class_<PySolver<T>, std::unique_ptr<PySolver<T>>>(m, ("OdeSolver" + suffix).c_str())
        .def_property_readonly("t", [](const PySolver<T>& self){return self->t();})
        .def_property_readonly("q", [](const PySolver<T>& self){return self->q();})
        .def_property_readonly("stepsize", [](const PySolver<T>& self){return self->stepsize();})
        .def_property_readonly("diverges", [](const PySolver<T>& self){return self->diverges();})
        .def_property_readonly("is_dead", [](const PySolver<T>& self){return self->is_dead();})
        .def_property_readonly("Nsys", [](const PySolver<T>& self){return self->Nsys();})
        .def("show_state", [](const PySolver<T>& self, int digits){
                return self->state().show(digits);
            },
            py::arg("digits") = 8
        )
        .def("advance", [](PySolver<T>& self){return self->advance();})
        .def("advance_to_event", [](PySolver<T>& self){return self->advance_to_event();})
        .def("reset", [](PySolver<T>& self){return self.s->reset();});

    py::class_<PyRK23<T>, PySolver<T>>(m, ("RK23" + suffix).c_str())
        .def(py::init<py::object, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple());


    py::class_<PyRK45<T>, PySolver<T>>(m, ("RK45" + suffix).c_str())
        .def(py::init<py::object, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple());

    py::class_<PyDOP853<T>, PySolver<T>>(m, ("DOP853" + suffix).c_str())
        .def(py::init<py::object, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple());


    py::class_<PyBDF<T>, PySolver<T>>(m, ("BDF" + suffix).c_str())
        .def(py::init<py::object, py::object, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
            py::arg("f"),
            py::arg("jac"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple());

    py::class_<PyOdeResult<T>>(m, ("OdeResult" + suffix).c_str())
        .def(py::init<PyOdeResult<T>>(), py::arg("result"))
        .def_property_readonly("t", [](const PyOdeResult<T>& self){
            return py::cast(NdView<const T>(self.res->t().data(), self.res->t().size()));
        })
        .def_property_readonly("q", [](const PyOdeResult<T>& self){
            return py::cast(NdView<const T>(self.res->q().data(), getShape(self.res->t().size(), self.q0_shape)));
        })
        .def_property_readonly("event_map", [](const PyOdeResult<T>& self){
            return to_PyDict(self.res->event_map());
        })
        .def("event_data", [](const PyOdeResult<T>& self, const py::str& event){
            std::vector<T> t_data = self.res->t_filtered(event.cast<std::string>());
            Array<T> t_res(t_data.data(), t_data.size());
            Array2D<T> q_data = self.res->q_filtered(event.cast<std::string>());
            return py::make_tuple(py::cast(t_res), py::cast(q_data));
        }, py::arg("event"))
        .def_property_readonly("diverges", [](const PyOdeResult<T>& self){return self.res->diverges();})
        .def_property_readonly("success", [](const PyOdeResult<T>& self){return self.res->success();})
        .def_property_readonly("runtime", [](const PyOdeResult<T>& self){return self.res->runtime();})
        .def_property_readonly("message", [](const PyOdeResult<T>& self){return self.res->message();})
        .def("examine", [](const PyOdeResult<T>& self){return self.res->examine();});

    py::class_<PyOdeSolution<T>, PyOdeResult<T>>(m, ("OdeSolution" + suffix).c_str())
        .def(py::init<PyOdeSolution<T>>(), py::arg("result"))
        .def("__call__", [](const PyOdeSolution<T>& self, const T& t){return self(t);})
        .def("__call__", [](const PyOdeSolution<T>& self, const py::iterable& array){return self(array);});


    py::class_<PyODE<T>>(m, ("LowLevelODE" + suffix).c_str())
        .def(py::init<py::object, T, py::iterable, py::object, T, T, T, T, T, int, py::iterable, py::iterable, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45")
        .def(py::init<PyODE<T>>(), py::arg("ode"))
        .def("solver", [](const PyODE<T>& self){return self.solver_copy();})
        .def("integrate", &PyODE<T>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("rich_integrate", &PyODE<T>::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("go_to", &PyODE<T>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("t_eval")=py::none(),
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0)
        .def("copy", [](const PyODE<T>& self){return PyODE<T>(self);})
        .def("reset", [](PyODE<T>& self){self.ode->reset();})
        .def("clear", [](PyODE<T>& self){self.ode->clear();})
        .def("event_data", [](const PyODE<T>& self, const py::str& event){
            std::vector<T> t_data = self.ode->t_filtered(event.cast<std::string>());
            Array<T> t_res(t_data.data(), t_data.size());
            Array2D<T> q_data = self.ode->q_filtered(event.cast<std::string>());
            return py::make_tuple(t_res, q_data);
        }, py::arg("event"))
        .def_property_readonly("Nsys", [](const PyODE<T>& self){return self.ode->Nsys();})
        .def_property_readonly("t", [](const PyODE<T>& self){return self.t_array();})
        .def_property_readonly("q", [](const PyODE<T>& self){return self.q_array();})
        .def_property_readonly("event_map", [](const PyODE<T>& self){return to_PyDict(self.ode->event_map());})
        .def_property_readonly("runtime", [](const PyODE<T>& self){return self.ode->runtime();})
        .def_property_readonly("diverges", [](const PyODE<T>& self){return self.ode->diverges();})
        .def_property_readonly("is_dead", [](const PyODE<T>& self){return self.ode->is_dead();});

    py::class_<PyVarODE<T>, PyODE<T>>(m, ("VariationalLowLevelODE" + suffix).c_str())
        .def(py::init<py::object, T, py::iterable, T, py::object, T, T, T, T, T, int, py::iterable, py::iterable, py::str>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=inf<T>(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45")
        .def(py::init<PyVarODE<T>>(), py::arg("ode"))
        .def_property_readonly("t_lyap", &PyVarODE<T>::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE<T>::py_lyap)
        .def_property_readonly("kicks", &PyVarODE<T>::py_kicks)
        .def("copy", [](const PyVarODE<T>& self){return PyVarODE<T>(self);});

        py::class_<PyFuncWrapper<T>>(m, ("LowLevelFunction" + suffix).c_str())
        .def(py::init<py::capsule, size_t, py::iterable, size_t>(), py::arg("pointer"), py::arg("input_size"), py::arg("output_shape"), py::arg("Nargs"))
        .def("__call__", &PyFuncWrapper<T>::call, py::arg("t"), py::arg("q"));
}


#endif
