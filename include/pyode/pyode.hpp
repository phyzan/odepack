#ifndef PYODE_HPP
#define PYODE_HPP


#include "../odepack/variational.hpp"
#include "pytools.hpp"


template<typename T>
StepSequence<T> to_step_sequence(const py::object& t_eval){
    if (t_eval.is_none()){
        return StepSequence<T>();
    }
    else if (py::iterable(t_eval)){
        std::vector<T> vec = toCPP_Array<T, std::vector<T>>(t_eval);
        return StepSequence<T>(vec.data(), vec.size());
    }
    else{
        throw py::type_error("Expected None, a NumPy array, or an iterable of numeric values.");
    }
}

inline py::dict to_PyDict(const EventMap& _map){
    py::dict py_dict;
    for (const auto& [key, vec] : _map) {
        py::array_t<size_t> np_array(static_cast<py::ssize_t>(vec.size()), vec.data()); // Create NumPy array
        py_dict[key.c_str()] = np_array; // Assign to dictionary
    }
    return py_dict;
}

template<typename T>
struct PyFuncWrapper{

    Func<T> rhs;
    size_t Nsys;
    std::vector<py::ssize_t> output_shape;
    size_t Nargs;
    size_t output_size;


    PyFuncWrapper(py::capsule obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs) : rhs(open_capsule<Func<T>>(obj)), Nsys(static_cast<size_t>(Nsys)), output_shape(static_cast<size_t>(output_shape.size())), Nargs(static_cast<size_t>(Nargs)) {
        copy_array(this->output_shape.data(), output_shape.data(), this->output_shape.size());
        long s = 1;
        for (long i : this->output_shape){
            s *= i;
        }
        this->output_size = s;
    }

    Array<T> call(const T& t, const std::vector<T>& py_q, py::args py_args) const {
        if (static_cast<size_t>(py_q.size()) != Nsys || py_args.size() != Nargs){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        auto args = toCPP_Array<T, std::vector<T>>(py_args);
        Array<T> res(output_shape);
        this->rhs(res.data(), t, py_q.data(), args.data(), nullptr);
        return res;
    }
};


template<typename T>
class PyEvent{

public:

    PyEvent(std::string name, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs) : _name(std::move(name)), _hide_mask(hide_mask), _Nsys(Nsys), _Nargs(Nargs){
        if (py::isinstance<py::capsule>(mask)){
            this->_mask = open_capsule<Func<T>>(mask);
        }
        else if (py::isinstance<py::function>(mask)){
            data.mask = mask;
            this->_mask = py_mask<T>;
        }
    }

    const std::string&  name() const{ return _name;}

    bool                hide_mask() const {return _hide_mask;}

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

    virtual std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) = 0;

    virtual bool equals(const PyEvent<T>& other) const{
        if (this == &other){
            return true;
        }
        return *this == other;
    }

    bool operator==(const PyEvent<T>& other) const = default;

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

    PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, T event_tol, size_t Nsys, size_t Nargs) : PyEvent<T>(name, mask, hide_mask, Nsys, Nargs), _dir(sgn(dir)), _event_tol(event_tol){
        if (py::isinstance<py::capsule>(when)){
            this->_when = open_capsule<ObjFun<T>>(when);
        }
        else if (py::isinstance<py::function>(when)){
            this->data.event = when;
            this->_when = py_event<T>;
        }

    }

    DEFAULT_RULE_OF_FOUR(PyPrecEvent);

    bool equals(const PyEvent<T>& other) const override{
        if (&other == this){
            return true;
        }
        else if (const auto* p = dynamic_cast<const PyPrecEvent<T>*>(&other)){
            return PyEvent<T>::operator==(*p) && (*this == *p);
        }
        return false;
    }

    T event_tol() const {return _event_tol;}

    std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override {
        if (this->is_lowlevel()){
            for (const py::handle& arg : args){
                if (PyNumber_Check(arg.ptr()) == 0){
                    throw py::value_error("All args must be numbers");
                }
            }
        }
        this->data.py_args = args;
        this->data.shape = shape;
        return std::make_unique<ObjectOwningEvent<PreciseEvent<T, 0>, PyStruct>>(this->data, this->name(), this->_when, _dir, this->_mask, this->hide_mask(), this->event_tol());
    }

    bool operator==(const PyPrecEvent<T>& other) const = default;

protected:

    int _dir = 0;
    T _event_tol;
    ObjFun<T> _when = nullptr;
};


template<typename T>
class PyPerEvent : public PyEvent<T>{

public:

    PyPerEvent(std::string name, T period, py::object start, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs):PyEvent<T>(name, mask, hide_mask, Nsys, Nargs), _period(period), _start(start.is_none() ? inf<T>() : start.cast<T>()){}

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    std::unique_ptr<Event<T, 0>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override {

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

    const T& period()const{
        return _period;
    }

    const T& start()const{
        return _start;
    }

    bool operator==(const PyPerEvent<T>& other) const = default;

    bool equals(const PyEvent<T>& other) const override{
        if (&other == this){
            return true;
        }
        else if (const auto* p = dynamic_cast<const PyPerEvent<T>*>(&other)){
            return PyEvent<T>::operator==(*p) && (*this == *p);
        }
        return false;
    }

private:
    T _period;
    T _start;

};


struct PyOptions : public EventOptions{

    PyOptions(std::string name, int max_events, bool terminate, int period) : EventOptions{.name=std::move(name), .max_events=max_events, .terminate=terminate, .period=period} {}

};

inline std::vector<EventOptions> to_Options(const py::iterable& d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        auto opt = py::cast<PyOptions>(item);
        result.emplace_back(opt);
    }
    result.shrink_to_fit();
    return result;
}


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


template<typename T>
OdeData<T> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);
    
    bool f_is_compiled = py::isinstance<PyFuncWrapper<T>>(f);
    bool jac_is_compiled = !jacobian.is_none() && py::isinstance<PyFuncWrapper<T>>(jacobian);
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<T> ode_rhs = {nullptr, nullptr, &data};
    if (f_is_compiled){
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
        data.rhs = f;
        ode_rhs.rhs = py_rhs;
    }
    if (jac_is_compiled){
        PyFuncWrapper<T>& _j = jacobian.cast<PyFuncWrapper<T>&>();
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
        if (!py::isinstance<PyEvent<T>>(ev)){
            throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class");
        }
        const PyEvent<T>& _ev = ev.cast<const PyEvent<T>&>();
        _ev.check_sizes(_size, args.size());

    }
    return ode_rhs;
}


template<typename T>
struct PySolver {

    PySolver(const py::function& f, const py::object& jac, T t0, py::iterable py_q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
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

    OdeSolver<T, 0>* operator->(){
        return s;
    }

    const OdeSolver<T, 0>* operator->() const{
        return s;
    }

    PySolver(const OdeSolver<T, 0>* other, PyStruct py_data) : s(other->clone()), data(std::move(py_data)){
        s->set_obj(&data);
    }

    PySolver(const PySolver& other) : s(other.s->clone()), data(other.data) {
        s->set_obj(&data);
    }

    PySolver(PySolver&& other) noexcept : s(other.s), data(std::move(other.data)) {
        other.s = nullptr;
        s->set_obj(&data);
    }

    PySolver& operator=(const PySolver& other){
        if (&other != this){
            delete s;
            s = other.s->clone();
            data = other.data;
            s->set_obj(&data);
        }
        return *this;
    }

    PySolver& operator=(PySolver&& other) noexcept{
        if (&other != this){
            delete s;
            s = other.s;
            other.s = nullptr;
            data = std::move(other.data);
            s->set_obj(&data);
        }
        return *this;
    }

    virtual ~PySolver(){
        delete s;
    }

    OdeSolver<T, 0>* s = nullptr;
    PyStruct data;
};

template<typename T>
struct PyRK23 : public PySolver<T>{

    PyRK23(const py::function& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK23"){}
};


template<typename T>
struct PyRK45 : public PySolver<T>{

    PyRK45(const py::function& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK45"){}
};

template<typename T>
struct PyDOP853 : public PySolver<T>{

    PyDOP853(const py::function& ode, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}
};


template<typename T>
struct PyBDF : public PySolver<T>{

    PyBDF(const py::function& f, const py::function& jac, T t0, py::iterable q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T>(f, jac, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}
};


template<typename T>
struct PyOdeResult{

    PyOdeResult(const OdeResult<T, 0>& r, const std::vector<py::ssize_t>& q0_shape): res(r.clone()), q0_shape(q0_shape){}

    PyOdeResult(const PyOdeResult& other) : res(other.res->clone()), q0_shape(other.q0_shape) {}

    PyOdeResult(PyOdeResult&& other) noexcept : res(other.res), q0_shape(std::move(other.q0_shape)) {
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

    PyOdeResult& operator=(PyOdeResult&& other) noexcept{
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
        Array<T> t_res(t_data.data());
        Array2D<T> q_data = this->res->q_filtered(event.cast<std::string>());
        return py::make_tuple(t_res, q_data);
    }

    virtual PyOdeResult<T>* clone() const{
        return new PyOdeResult<T>(*this);
    }

    OdeResult<T, 0>* res;
    std::vector<py::ssize_t> q0_shape;

};

template<typename T>
struct PyOdeSolution : public PyOdeResult<T>{

    PyOdeSolution(const OdeSolution<T, 0>& res, const std::vector<py::ssize_t>& q0_shape) : PyOdeResult<T>(res, q0_shape), nsys(prod(q0_shape)) {}

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    inline const OdeSolution<T, 0>& sol() const{
        return static_cast<const OdeSolution<T, 0>&>(*this->res);
    }

    inline Array<T> operator()(const T& t) const{
        auto res = this->sol()(t);
        return Array<T>(res.data(), this->q0_shape);
    }

    Array<T> operator()(const py::array& py_array) const{
        const auto nt = size_t(py_array.size());
        std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
        final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
        Array<T> res(final_shape);
        const T* data = static_cast<const T*>(py_array.data());
        for (size_t i=0; i<nt; i++){
            copy_array(res.data()+i*nsys, this->sol()(data[i]).data(), nsys);
        }
        return res;
    }

    PyOdeResult<T>* clone() const override{
        return new PyOdeSolution<T>(*this);
    }

    size_t nsys;

};



template<typename T>
class PyODE{
    
public:

    PyODE(py::function f, T t0, py::iterable py_q0, py::object jacobian, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method){
        
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

protected:
    PyODE() = default;//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE<T>& other) : ode(other.ode->clone()), data(other.data){
        ode->set_obj(&data);
    }

    PyODE(PyODE<T>&& other) noexcept :ode(other.ode), data(std::move(other.data)){
        other.ode = nullptr;
        ode->set_obj(&data);
    }

    PyODE<T>& operator=(const PyODE<T>& other){
        if (&other == this){
            return *this;
        }
        delete ode;
        ode = other.ode->clone();
        data = other.data;
        ode->set_obj(&data);
        return *this;
    }

    PyODE<T>& operator=(PyODE<T>&& other) noexcept {
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

    virtual ~PyODE(){
        delete ode;
    }

    PyOdeResult<T> py_integrate(const T& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){
        OdeResult<T, 0> res = ode->integrate(interval, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
        PyOdeResult<T> py_res(res, data.shape);
        return py_res;
    }

    PyOdeSolution<T> py_rich_integrate(const T& interval, const py::iterable& event_options, int max_prints){
        OdeSolution<T, 0> res = ode->rich_integrate(interval, to_Options(event_options), max_prints);
        PyOdeSolution<T> py_res(res, data.shape);
        return py_res;
    }

    PyOdeResult<T> py_go_to(const T& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){
        OdeResult<T, 0> res = ode->go_to(t, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
        PyOdeResult<T> py_res(res, data.shape);
        return py_res;
    }

    NdView<const T> t_array()const{
        return NdView<const T>(ode->t().data(), ode->t().size());
    }

    NdView<const T, Layout::C, 0, 0> q_array()const{
        return ode->q();
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = ode->t_filtered(event.cast<std::string>());
        Array2D<T, 0, 0> q_data = ode->q_filtered(event.cast<std::string>());
        Array<T> q_res(q_data.data(), getShape(q_data.shape(0), data.shape), true);
        return py::make_tuple(t_data, q_res);
    }

    PySolver<T> solver_copy() const{
        return PySolver<T>(ode->solver(), data);
    }

    ODE<T, 0>* ode = nullptr;
    PyStruct data;
};


template<typename T>
class PyVarODE : public PyODE<T>{

public:

    PyVarODE(py::function f, T t0, py::iterable q0, T period, py::object jac, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method):PyODE<T>(){
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

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    VariationalODE<T, 0>& varode(){
        return *static_cast<VariationalODE<T, 0>*>(this->ode);
    }

    const VariationalODE<T, 0>& varode() const {
        return *static_cast<VariationalODE<T, 0>*>(this->ode);
    }

    NdView<const T> py_t_lyap() const{
        return NdView<const T>(varode().t_lyap().data(), varode().t_lyap().size());
    }

    NdView<const T> py_lyap() const{
        return NdView<const T>(varode().lyap().data(), varode().t_lyap().size());
    }

    NdView<const T> py_kicks() const{
        return NdView<const T>(varode().kicks().data(), varode().t_lyap().size());
    }

};

template<typename T>
void define_ode_module(py::module& m){

    py::class_<PyOptions>(m, "EventOpt")
        .def(py::init<py::str, int, bool, int>(), py::arg("name"), py::arg("max_events")=-1, py::arg("terminate")=false, py::arg("period")=1);
        
    py::class_<PyEvent<T>>(m, "Event")
        .def_property_readonly("name", [](const PyEvent<T>& self){
            return self.name();
        })
        .def_property_readonly("hides_mask", [](const PyEvent<T>& self){
            return self.hide_mask();
        })
        .def("__eq__", [](const PyEvent<T>& self, const PyEvent<T>& other){return self.equals(other);});

    py::class_<PyPrecEvent<T>, PyEvent<T>>(m, "PreciseEvent")
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

    py::class_<PyPerEvent<T>, PyEvent<T>>(m, "PeriodicEvent")
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

    py::class_<PySolver<T>, std::unique_ptr<PySolver<T>>>(m, "OdeSolver")
        .def_property_readonly("t", [](const PySolver<T>& self){return py::cast(self->t());})
        .def_property_readonly("q", [](const PySolver<T>& self){return py::cast(self->q());})
        .def_property_readonly("stepsize", [](const PySolver<T>& self){return self->stepsize();})
        .def_property_readonly("diverges", [](const PySolver<T>& self){return self->diverges();})
        .def_property_readonly("is_dead", [](const PySolver<T>& self){return self->is_dead();})
        .def_property_readonly("Nsys", [](const PySolver<T>& self){return self->Nsys();})
        .def("show_state", [](const PySolver<T>& self, int digits = 8){return self->state().show(digits);})
        .def("advance", [](PySolver<T>& self){return self->advance();})
        .def("advance_to_event", [](PySolver<T>& self){return self->advance_to_event();})
        .def("reset", [](PySolver<T>& self){return self.s->reset();});

    py::class_<PyRK23<T>, PySolver<T>>(m, "RK23")
        .def(py::init<py::function, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
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

    
    py::class_<PyRK45<T>, PySolver<T>>(m, "RK45")
        .def(py::init<py::function, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
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

    py::class_<PyDOP853<T>, PySolver<T>>(m, "DOP853")
        .def(py::init<py::function, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
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


    py::class_<PyBDF<T>, PySolver<T>>(m, "BDF")
        .def(py::init<py::function, py::function, T, py::iterable, T, T, T, T, T, int, py::iterable, py::iterable>(),
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
    
    py::class_<PyOdeResult<T>>(m, "OdeResult")
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
    
    py::class_<PyOdeSolution<T>, PyOdeResult<T>>(m, "OdeSolution")
        .def(py::init<PyOdeSolution<T>>(), py::arg("result"))
        .def("__call__", [](const PyOdeSolution<T>& self, const T& t){return self(t);})
        .def("__call__", [](const PyOdeSolution<T>& self, const py::iterable& array){return self(array);});


    py::class_<PyODE<T>>(m, "LowLevelODE")
        .def(py::init<py::function, T, py::iterable, py::object, T, T, T, T, T, int, py::iterable, py::iterable, py::str>(),
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
            return py::make_tuple(py::cast(t_res), py::cast(q_data));
        }, py::arg("event"))
        .def_property_readonly("Nsys", [](const PyODE<T>& self){return self.ode->Nsys();})
        .def_property_readonly("t", [](const PyODE<T>& self){return py::cast(self.t_array());})
        .def_property_readonly("q", [](const PyODE<T>& self){return py::cast(self.q_array());})
        .def_property_readonly("event_map", [](const PyODE<T>& self){return to_PyDict(self.ode->event_map());})
        .def_property_readonly("runtime", [](const PyODE<T>& self){return self.ode->runtime();})
        .def_property_readonly("diverges", [](const PyODE<T>& self){return self.ode->diverges();})
        .def_property_readonly("is_dead", [](const PyODE<T>& self){return self.ode->is_dead();});

    py::class_<PyVarODE<T>, PyODE<T>>(m, "VariationalLowLevelODE")
        .def(py::init<py::function, T, py::iterable, T, py::object, T, T, T, T, T, int, py::iterable, py::iterable, py::str>(),
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
    
    py::class_<PyFuncWrapper<T>>(m, "LowLevelFunction")
        .def(py::init<py::capsule, size_t, py::iterable, size_t>(), py::arg("pointer"), py::arg("input_size"), py::arg("output_shape"), py::arg("Nargs"))
        .def("__call__", &PyFuncWrapper<T>::call, py::arg("t"), py::arg("q"));

    m.def("integrate_all", [](const py::object& list, const T& interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress){
            std::vector<ODE<T, 0>*> array;
            for (const py::handle& item : list) {
                try {
                    // Attempt the cast
                    PyODE<T>& pyode = item.cast<PyODE<T>&>();
                    array.push_back(pyode.ode);
                } catch (const py::cast_error& e) {
                    throw py::value_error("List item is not a PyODE object of the expected type.");
                }
            }
            integrate_all(array, interval, to_step_sequence<T>(t_eval), to_Options(event_options), threads, display_progress);
        }, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);
}



#endif