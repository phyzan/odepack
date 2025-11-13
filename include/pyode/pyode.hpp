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

template<typename T, size_t N>
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

    py::array_t<T> call(const T& t, const py::array_t<T>& py_q, py::args py_args) const {
        if (static_cast<size_t>(py_q.size()) != Nsys || py_args.size() != Nargs){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        auto args = toCPP_Array<T, std::vector<T>>(py_args);
        T* res = new T[output_size];
        this->rhs(res, t, py_q.data(), args.data(), nullptr);
        return array(res, output_shape);
    }
};


template<typename T, size_t N>
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

    virtual std::unique_ptr<Event<T, N>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) = 0;

    virtual bool equals(const PyEvent<T, N>& other) const{
        if (this == &other){
            return true;
        }
        return *this == other;
    }

    bool operator==(const PyEvent<T, N>& other) const = default;

    virtual ~PyEvent(){}

protected:

    std::string _name;
    bool _hide_mask;

    Func<T> _mask = nullptr;

    size_t _Nsys;
    size_t _Nargs;

    PyStruct data;
};


template<typename T, size_t N>
class PyPrecEvent : public PyEvent<T, N> {

public:

    PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, T event_tol, size_t Nsys, size_t Nargs) : PyEvent<T, N>(name, mask, hide_mask, Nsys, Nargs), _dir(sgn(dir)), _event_tol(event_tol){
        if (py::isinstance<py::capsule>(when)){
            this->_when = open_capsule<ObjFun<T>>(when);
        }
        else if (py::isinstance<py::function>(when)){
            this->data.event = when;
            this->_when = py_event<T>;
        }

    }

    DEFAULT_RULE_OF_FOUR(PyPrecEvent);

    bool equals(const PyEvent<T, N>& other) const override{
        if (&other == this){
            return true;
        }
        else if (const auto* p = dynamic_cast<const PyPrecEvent<T, N>*>(&other)){
            return PyEvent<T, N>::operator==(*p) && (*this == *p);
        }
        return false;
    }

    T event_tol() const {return _event_tol;}

    std::unique_ptr<Event<T, N>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override {
        if (this->is_lowlevel()){
            for (const py::handle& arg : args){
                if (PyNumber_Check(arg.ptr()) == 0){
                    throw py::value_error("All args must be numbers");
                }
            }
        }
        this->data.py_args = args;
        this->data.shape = shape;
        return std::make_unique<ObjectOwningEvent<PreciseEvent<T, N>, PyStruct>>(this->data, this->name(), this->_when, _dir, this->_mask, this->hide_mask(), this->event_tol());
    }

    bool operator==(const PyPrecEvent<T, N>& other) const = default;

protected:

    int _dir = 0;
    T _event_tol;
    ObjFun<T> _when = nullptr;
};


template<typename T, size_t N>
class PyPerEvent : public PyEvent<T, N>{

public:

    PyPerEvent(std::string name, T period, py::object start, py::object mask, bool hide_mask, size_t Nsys, size_t Nargs):PyEvent<T, N>(name, mask, hide_mask, Nsys, Nargs), _period(period), _start(start.is_none() ? inf<T>() : start.cast<T>()){}

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    std::unique_ptr<Event<T, N>> toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override {

        if (this->is_lowlevel()){
            for (const py::handle& arg : args){
                if (PyNumber_Check(arg.ptr())==0){
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

    bool operator==(const PyPerEvent<T, N>& other) const = default;

    bool equals(const PyEvent<T, N>& other) const override{
        if (&other == this){
            return true;
        }
        else if (const auto* p = dynamic_cast<const PyPerEvent<T, N>*>(&other)){
            return PyEvent<T, N>::operator==(*p) && (*this == *p);
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


template<typename T, size_t N>
std::vector<std::unique_ptr<Event<T, N>>> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, py::iterable args){   
    if (events.is_none()){
        return {};
    }
    std::vector<std::unique_ptr<Event<T, N>>> res;
    for (py::handle item : events){
        res.push_back(item.cast<PyEvent<T, N>&>().toEvent(shape, args));
    }
    return res;
}


template<typename T, size_t N>
OdeData<T> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::array_t<T>& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);
    
    bool f_is_compiled = py::isinstance<PyFuncWrapper<T, N>>(f);
    bool jac_is_compiled = !jacobian.is_none() && py::isinstance<PyFuncWrapper<T, N>>(jacobian);
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<T> ode_rhs = {nullptr, nullptr, &data};
    if (f_is_compiled){
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
    if (jac_is_compiled){
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


template<typename T, size_t N>
struct PySolver {

    PySolver(const py::function& f, const py::object& jac, T t0, py::array_t<T> q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
        std::vector<T> args;
        OdeData<T> ode_data = init_ode_data<T, N>(this->data, args, f, q0, jac, py_args, py_events);
        auto safe_events = to_Events<T, N>(py_events, this->data.shape, py_args);
        std::vector<const Event<T, N>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i].get();
        }
        this->s = get_solver(name, ode_data, t0, Array1D<T, N>(q0.data(), static_cast<size_t>(q0.size())), rtol, atol, min_step, max_step, first_step, dir, args, evs).release();
    }

    OdeSolver<T, N>* operator->(){
        return s;
    }

    const OdeSolver<T, N>* operator->() const{
        return s;
    }

    PySolver(const OdeSolver<T, N>* other, PyStruct py_data) : s(other->clone()), data(std::move(py_data)){
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

    OdeSolver<T, N>* s = nullptr;
    PyStruct data;
};

template<typename T, size_t N>
struct PyRK23 : public PySolver<T, N>{

    PyRK23(const py::function& ode, T t0, py::array_t<T> q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T, N>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK23"){}
};


template<typename T, size_t N>
struct PyRK45 : public PySolver<T, N>{

    PyRK45(const py::function& ode, T t0, py::array_t<T> q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T, N>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK45"){}
};

template<typename T, size_t N>
struct PyDOP853 : public PySolver<T, N>{

    PyDOP853(const py::function& ode, T t0, py::array_t<T> q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T, N>(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}
};


template<typename T, size_t N>
struct PyBDF : public PySolver<T, N>{

    PyBDF(const py::function& f, const py::function& jac, T t0, py::array_t<T> q0, T rtol, T atol, T min_step, T max_step, T first_step, int dir, const py::iterable& args, const py::iterable& events) : PySolver<T, N>(f, jac, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853"){}
};


template<typename T, size_t N>
struct PyOdeResult{

    PyOdeResult(const OdeResult<T, N>& r, const std::vector<py::ssize_t>& q0_shape): res(r.clone()), q0_shape(q0_shape){}

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
        py::array_t<T> t_res = to_numpy<T>(t_data);
        Array2D<T, 0, N> q_data = this->res->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(q_data, getShape(t_data.size(), this->q0_shape));
        return py::make_tuple(t_res, q_res);
    }

    virtual PyOdeResult<T, N>* clone() const{
        return new PyOdeResult<T, N>(*this);
    }

    OdeResult<T, N>* res;
    std::vector<py::ssize_t> q0_shape;

};

template<typename T, size_t N>
struct PyOdeSolution : public PyOdeResult<T, N>{

    PyOdeSolution(const OdeSolution<T, N>& res, const std::vector<py::ssize_t>& q0_shape) : PyOdeResult<T, N>(res, q0_shape), nsys(prod(q0_shape)) {}

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    inline const OdeSolution<T, N>& sol() const{
        return static_cast<const OdeSolution<T, N>&>(*this->res);
    }

    inline py::array_t<T> operator()(const T& t) const{
        return to_numpy<T>(this->sol()(t), this->q0_shape);
    }

    py::array_t<T> operator()(const py::array_t<T>& py_array) const{
        const size_t nt = py_array.size();
        std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
        final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
        T* res = new T[nt*nsys];
        const T* data = py_array.data();
        for (size_t i=0; i<nt; i++){
            copy_array(res+i*nsys, this->sol()(data[i]).data(), nsys);
        }
        return array(res, final_shape);
    }

    PyOdeResult<T, N>* clone() const override{
        return new PyOdeSolution<T, N>(*this);
    }

    size_t nsys;

};



template<typename T, int N>
class PyODE{

public:

    PyODE(py::function f, T t0, py::array_t<T> q0, py::object jacobian, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method){
        std::vector<T> args;
        OdeData<T> ode_rhs = init_ode_data<T, N>(data,args, f, q0, jacobian, py_args, events);
        auto safe_events = to_Events<T, N>(events, shape(q0), py_args);
        std::vector<const Event<T, N>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i].get();
        }
        ode = new ODE<T, N>(ode_rhs, t0, toCPP_Array<T, Array1D<T, N>>(q0), rtol, atol, min_step, max_step, first_step, dir, args, evs, method);
    }

protected:
    PyODE() = default;//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE<T, N>& other) : ode(other.ode->clone()), data(other.data){
        ode->set_obj(&data);
    }

    PyODE(PyODE<T, N>&& other) noexcept :ode(other.ode), data(std::move(other.data)){
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

    PyODE<T, N>& operator=(PyODE<T, N>&& other) noexcept {
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

    PyOdeResult<T, N> py_integrate(const T& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){
        OdeResult<T, N> res = ode->integrate(interval, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
        PyOdeResult<T, N> py_res(res, data.shape);
        return py_res;
    }

    PyOdeSolution<T, N> py_rich_integrate(const T& interval, const py::iterable& event_options, int max_prints){
        OdeSolution<T, N> res = ode->rich_integrate(interval, to_Options(event_options), max_prints);
        PyOdeSolution<T, N> py_res(res, data.shape);
        return py_res;
    }

    PyOdeResult<T, N> py_go_to(const T& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){
        OdeResult<T, N> res = ode->go_to(t, to_step_sequence<T>(t_eval), to_Options(event_options), max_prints);
        PyOdeResult<T, N> py_res(res, data.shape);
        return py_res;
    }

    py::array_t<T> t_array()const{
        return to_numpy<T>(ode->t());
    }

    py::array_t<T> q_array()const{
        return to_numpy<T>(ode->q(), getShape(ode->t().size(), data.shape));
    }

    py::tuple event_data(const py::str& event)const{
        std::vector<T> t_data = ode->t_filtered(event.cast<std::string>());
        py::array_t<T> t_res = to_numpy<T>(t_data);
        Array2D<T, 0, N> q_data = ode->q_filtered(event.cast<std::string>());
        py::array_t<T> q_res = to_numpy<T>(q_data, getShape(q_data.shape(0), data.shape));
        return py::make_tuple(t_res, q_res);
    }

    PySolver<T, N> solver_copy() const{
        return PySolver<T, N>(ode->solver(), data);
    }

    ODE<T, N>* ode = nullptr;
    PyStruct data;
};


template<typename T, size_t N>
class PyVarODE : public PyODE<T, N>{

public:

    PyVarODE(py::function f, T t0, py::iterable q0, T period, py::object jac, T rtol, T atol, T min_step, T max_step, T first_step, int dir, py::iterable py_args, py::iterable events, const py::str& method):PyODE<T, N>(){
        std::vector<T> args;
        OdeData<T> ode_rhs = init_ode_data<T, N>(this->data, args, f, q0, jac, py_args, events);
        Array1D<T, N> q0_ = toCPP_Array<T, Array1D<T, N>>(q0);
        auto safe_events = to_Events<T, N>(events, shape(q0), py_args);
        std::vector<const Event<T, N>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i].get();
        }
        this->ode = new VariationalODE<T, N>(ode_rhs, t0, q0_, period, rtol, atol, min_step, max_step, first_step, dir, args, evs, method.cast<std::string>());
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

    py::array_t<T> py_kicks() const{
        return to_numpy<T>(varode().kicks());
    }

};

template<typename T, size_t N>
void define_ode_module(py::module& m){

    py::class_<PyOptions>(m, "EventOpt")
        .def(py::init<py::str, int, bool, int>(), py::arg("name"), py::arg("max_events")=-1, py::arg("terminate")=false, py::arg("period")=1);
        
    py::class_<PyEvent<T, N>>(m, "Event")
        .def_property_readonly("name", [](const PyEvent<T, N>& self){
            return self.name();
        })
        .def_property_readonly("hides_mask", [](const PyEvent<T, N>& self){
            return self.hide_mask();
        })
        .def("__eq__", [](const PyEvent<T, N>& self, const PyEvent<T, N>& other){return self.equals(other);});

    py::class_<PyPrecEvent<T, N>, PyEvent<T, N>>(m, "PreciseEvent")
        .def(py::init<std::string, py::object, int, py::object, bool, T, size_t, size_t>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("dir")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12,
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("event_tol", [](const PyPrecEvent<T, N>& self){
            return self.event_tol();
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

    py::class_<PySolver<T, N>, std::unique_ptr<PySolver<T, N>>>(m, "OdeSolver")
        .def_property_readonly("t", [](const PySolver<T, N>& self){return self->t();})
        .def_property_readonly("q", [](const PySolver<T, N>& self){return to_numpy<T>(self->q(), self.data.shape);})
        .def_property_readonly("stepsize", [](const PySolver<T, N>& self){return self->stepsize();})
        .def_property_readonly("diverges", [](const PySolver<T, N>& self){return self->diverges();})
        .def_property_readonly("is_dead", [](const PySolver<T, N>& self){return self->is_dead();})
        .def_property_readonly("Nsys", [](const PySolver<T, N>& self){return self->Nsys();})
        .def("show_state", [](const PySolver<T, N>& self){return self->state().show();})
        .def("advance", [](PySolver<T, N>& self){return self->advance();})
        .def("advance_to_event", [](PySolver<T, N>& self){return self->advance_to_event();})
        .def("reset", [](PySolver<T, N>& self){return self.s->reset();});

    py::class_<PyRK23<T, N>, PySolver<T, N>>(m, "RK23")
        .def(py::init<py::function, T, py::array_t<T>, T, T, T, T, T, int, py::iterable, py::iterable>(),
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

    
    py::class_<PyRK45<T, N>, PySolver<T, N>>(m, "RK45")
        .def(py::init<py::function, T, py::array_t<T>, T, T, T, T, T, int, py::iterable, py::iterable>(),
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

    py::class_<PyDOP853<T, N>, PySolver<T, N>>(m, "DOP853")
        .def(py::init<py::function, T, py::array_t<T>, T, T, T, T, T, int, py::iterable, py::iterable>(),
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


    py::class_<PyBDF<T, N>, PySolver<T, N>>(m, "BDF")
        .def(py::init<py::function, py::function, T, py::array_t<T>, T, T, T, T, T, int, py::iterable, py::iterable>(),
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
    
    py::class_<PyOdeResult<T, N>>(m, "OdeResult")
        .def(py::init<PyOdeResult<T, N>>(), py::arg("result"))
        .def_property_readonly("t", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(self.res->t());
        })
        .def_property_readonly("q", [](const PyOdeResult<T, N>& self){
            return to_numpy<T>(self.res->q(), getShape(self.res->t().size(), self.q0_shape));
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


    py::class_<PyODE<T, N>>(m, "LowLevelODE")
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
        .def(py::init<PyODE<T, N>>(), py::arg("ode"))
        .def("solver", [](const PyODE<T, N>& self){return self.solver_copy();})
        .def("integrate", &PyODE<T, N>::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("rich_integrate", &PyODE<T, N>::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("go_to", &PyODE<T, N>::py_go_to,
        py::arg("t"),
        py::kw_only(),
        py::arg("t_eval")=py::none(),
        py::arg("event_options")=py::tuple(),
        py::arg("max_prints")=0)
        .def("copy", [](const PyODE<T, N>& self){return PyODE<T, N>(self);})
        .def("reset", [](PyODE<T, N>& self){self.ode->reset();})
        .def("clear", [](PyODE<T, N>& self){self.ode->clear();})
        .def("event_data", &PyODE<T, N>::event_data, py::arg("event"))
        .def_property_readonly("Nsys", [](const PyODE<T, N>& self){return self.ode->Nsys();})
        .def_property_readonly("t", &PyODE<T, N>::t_array)
        .def_property_readonly("q", &PyODE<T, N>::q_array)
        .def_property_readonly("event_map", [](const PyODE<T, N>& self){return to_PyDict(self.ode->event_map());})
        .def_property_readonly("runtime", [](const PyODE<T, N>& self){return self.ode->runtime();})
        .def_property_readonly("diverges", [](const PyODE<T, N>& self){return self.ode->diverges();})
        .def_property_readonly("is_dead", [](const PyODE<T, N>& self){return self.ode->is_dead();});

    py::class_<PyVarODE<T, N>, PyODE<T, N>>(m, "VariationalLowLevelODE")
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
        .def(py::init<PyVarODE<T, N>>(), py::arg("ode"))
        .def_property_readonly("t_lyap", &PyVarODE<T, N>::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE<T, N>::py_lyap)
        .def_property_readonly("kicks", &PyVarODE<T, N>::py_kicks)
        .def("copy", [](const PyVarODE<T, N>& self){return PyVarODE<T, N>(self);});
    
    py::class_<PyFuncWrapper<T, N>>(m, "LowLevelFunction")
        .def(py::init<py::capsule, size_t, py::iterable, size_t>(), py::arg("pointer"), py::arg("input_size"), py::arg("output_shape"), py::arg("Nargs"))
        .def("__call__", &PyFuncWrapper<T, N>::call, py::arg("t"), py::arg("q"));

    m.def("integrate_all", [](const py::object& list, const T& interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress){
            std::vector<ODE<T, N>*> array;
            for (const py::handle& item : list) {
                array.push_back(item.cast<PyODE<T, N>&>().ode);
            }
            integrate_all(array, interval, to_step_sequence<T>(t_eval), to_Options(event_options), threads, display_progress);
        }, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);
}



#endif