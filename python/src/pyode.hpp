#ifndef PYODE_HPP
#define PYODE_HPP

#include "pytools.hpp"

#define EXECUTE_ANY(DTYPE, ACTION, CppType, EXPR, DEFAULT)  \
    switch (DTYPE) {                                                         \
        case 0: ACTION CppType<double>EXPR break;              \
        case 1: ACTION CppType<long double>EXPR break;         \
        case 2: ACTION CppType<mpfr::mpreal>EXPR break;        \
        case 3: ACTION CppType<float>EXPR break;               \
        default: DEFAULT;   \
    }

#define EXECUTE(ACTION, CppType, EXPR, DEFAULT) EXECUTE_ANY(this->scalar_type, ACTION, CppType, EXPR, DEFAULT) \

#define PY_GET_TEMPLATE(TEMPLATE, BODY)                                      \
    switch (this->scalar_type) {                                                         \
        case 0: return py::cast(TEMPLATE<double>BODY); break;              \
        case 1: return py::cast(TEMPLATE<long double>BODY); break;         \
        case 2: return py::cast(TEMPLATE<mpfr::mpreal>BODY); break;        \
        case 3: return py::cast(TEMPLATE<float>BODY); break;               \
        default: return py::none();   \
    }

#define PY_GET(TEMPLATE, PTR, BODY) PY_GET_TEMPLATE(reinterpret_cast<TEMPLATE, *>(PTR)BODY)

#define PY_DO(TEMPLATE, PTR, BODY)                                      \
    switch (this->scalar_type) {                                                         \
        case 0: reinterpret_cast<TEMPLATE<double>*>(PTR)->BODY; break;              \
        case 1: reinterpret_cast<TEMPLATE<long double>*>(PTR)->BODY; break;         \
        case 2: reinterpret_cast<TEMPLATE<mpfr::mpreal>*>(PTR)->BODY; break;        \
        case 3: reinterpret_cast<TEMPLATE<float>*>(PTR)->BODY; break;               \
        default: ;   \
    }


static const std::map<std::string, int> DTYPE_MAP = {
    {"double", 0},
    {"long double", 1},
    {"mpreal", 2},
    {"float", 3}
};

static const std::string SCALAR_TYPE[4] = {"double", "long double", "mpreal", "float"};

template<typename T>
std::string get_scalar_type(){
    if constexpr (std::is_same_v<T, double>){
        return "double";
    }
    else if constexpr (std::is_same_v<T, long double>){
        return "long double";
    }
    else if constexpr (std::is_same_v<T, float>){
        return "float";
    }
    else if constexpr (std::is_same_v<T, mpfr::mpreal>){
        return "mpreal";
    }
    else{
        static_assert(false, "Unsupported scalar type T");
    }
}

/*
FORWARD DECLARATIONS & INTERFACES
*/


struct DtypeDispatcher{

    DtypeDispatcher(const std::string& dtype_){
        this->scalar_type = DTYPE_MAP.at(dtype_);
    }

    DtypeDispatcher(int dtype_) : scalar_type(dtype_){}

    int scalar_type;
};

struct PyFuncWrapper : DtypeDispatcher {

    void* rhs; //Func<T>
    size_t Nsys;
    std::vector<py::ssize_t> output_shape;
    size_t Nargs;
    size_t output_size;

    PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type);

    template<typename T>
    py::object call_impl(const py::object& t, const py::iterable& py_q, py::args py_args) const;

    py::object call(const py::object& t, const py::iterable& py_q, const py::args& py_args) const;
};


class PyEvent : public DtypeDispatcher{

public:

    PyEvent(std::string name, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    py::str             name() const;

    py::bool_           hide_mask() const;

    bool                is_lowlevel() const;

    void                check_sizes(size_t Nsys, size_t Nargs) const;

    template<typename T>
    Func<T> mask() const;

    virtual void*       toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) = 0;

    virtual ~PyEvent() = default;

protected:

    std::string _name;
    bool _hide_mask;
    void* _mask = nullptr;
    size_t _Nsys;
    size_t _Nargs;
    PyStruct data;
    bool _py_mask = false; //if true, cast _mask to py_mask<T>
};


class PyPrecEvent : public PyEvent {

public:

    PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, py::object event_tol, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPrecEvent);

    py::object event_tol() const;

    void* toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

    template<typename T>
    ObjFun<T> when() const;

    template<typename T>
    void* get_new_event();

protected:

    int _dir = 0;
    py::object _event_tol;
    void* _when = nullptr; //type: ObjFun<T>. If null, set it to py_event<T>
};



class PyPerEvent : public PyEvent{

public:

    PyPerEvent(std::string name, py::object period, py::object start, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    void* toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

    template<typename T>
    void* get_new_event();

    py::object period() const;

    py::object start() const;

private:
    py::object _period;
    py::object _start;

};


struct PySolver : DtypeDispatcher {

    PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type);

    PySolver(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}

    PySolver(void* solver, PyStruct py_data, int scalar_type);

    PySolver(const PySolver& other);

    PySolver(PySolver&& other) noexcept;

    PySolver& operator=(const PySolver& other);

    PySolver& operator=(PySolver&& other) noexcept;

    virtual ~PySolver();

    py::object t() const{
        PY_GET(OdeSolver, this->s, ->t())
    }

    py::object q() const{
        PY_GET(OdeSolver, this->s, ->q())
    }

    py::object stepsize() const{
        PY_GET(OdeSolver, this->s, ->stepsize())
    }

    py::object diverges() const{
        PY_GET(OdeSolver, this->s, ->diverges())
    }

    py::object is_dead() const{
        PY_GET(OdeSolver, this->s, ->is_dead())
    }

    py::object Nsys() const{
        PY_GET(OdeSolver, this->s, ->Nsys())
    }

    void show_state(int digits) const{
        PY_DO(OdeSolver, this->s, state().show(digits))
    }

    py::object advance() const{
        PY_GET(OdeSolver, this->s, ->advance())
    }

    py::object advance_to_event() const{
        PY_GET(OdeSolver, this->s, ->advance_to_event())
    }

    virtual py::object copy() const = 0;

    void reset() const{
        PY_DO(OdeSolver, this->s, reset())
    }

    template<typename T>
    void init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name);

    void* s = nullptr; //OdeSolver<T, 0>*
    PyStruct data;
};


struct PyRK23 : public PySolver{

    PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK23(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    py::object copy() const override;

};


struct PyRK45 : public PySolver{

    PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK45(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    py::object copy() const override;

};


struct PyDOP853 : public PySolver{

    PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyDOP853(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    py::object copy() const override;

};


struct PyBDF : public PySolver{

    PyBDF(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyBDF(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    py::object copy() const override;

};


struct PyOdeResult : DtypeDispatcher{

    PyOdeResult(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type);

    PyOdeResult(const PyOdeResult& other);

    PyOdeResult(PyOdeResult&& other) noexcept;

    PyOdeResult& operator=(const PyOdeResult& other);

    PyOdeResult& operator=(PyOdeResult&& other) noexcept;

    virtual ~PyOdeResult();

    py::object t() const{
        EXECUTE(return, this->_t, ();, return py::none();)
    }

    py::object q() const{
        EXECUTE(return, this->_q, ();, return py::none();)
    }

    py::dict event_map() const{
        EXECUTE(return, this->_event_map, ();, return py::none();)
    }

    py::tuple event_data(const py::str& event) const{
        EXECUTE(return, this->_event_data, (event);, return py::none();)
    }


    py::bool_ diverges() const{
        PY_GET(OdeResult, this->res, ->diverges())
    }

    py::bool_ success() const{
        PY_GET(OdeResult, this->res, ->success())
    }

    py::float_ runtime() const{
        PY_GET(OdeResult, this->res, ->runtime())
    }

    py::str message() const{
        PY_GET(OdeResult, this->res, ->message())
    }

    void examine() const{
        PY_DO(OdeResult, this->res, examine())
    }

    template<typename T>
    py::tuple _event_data(const py::str& event) const;

    template<typename T>
    py::dict _event_map() const{
        EventMap result = reinterpret_cast<const OdeResult<T>*>(this->res)->event_map();
        return to_PyDict(result);
    }

    template<typename T>
    py::object _t() const{
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        return py::cast(NdView<const T>(r->t().data(), r->t().size()));
    }

    template<typename T>
    py::object _q() const{
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        return py::cast(NdView<const T>(r->q().data(), getShape(py::ssize_t(r->t().size()), this->q0_shape)));
    }

    void* res = nullptr;
    std::vector<py::ssize_t> q0_shape;

};


struct PyOdeSolution : public PyOdeResult{

    PyOdeSolution(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    py::object operator()(const py::object& t) const;

    template<typename T>
    py::object _get_frame(const py::object& t) const;

    template<typename T>
    py::object _get_array(const py::array& py_array) const;

    size_t nsys;

};



class PyODE : public DtypeDispatcher{

public:

    PyODE(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

protected:

    PyODE(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE& other);

    PyODE(PyODE&& other) noexcept;

    PyODE& operator=(const PyODE& other);

    PyODE& operator=(PyODE&& other) noexcept;

    virtual ~PyODE();

    py::object py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints);

    py::object py_go_to(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object t_array() const;

    py::object q_array() const;

    py::tuple event_data(const py::str& event) const;

    virtual py::object copy() const;

    py::object solver_copy() const;

    py::dict event_map() const{
        EXECUTE(return, this->_event_map, ();, return py::none();)
    }

    py::object Nsys() const{
        PY_GET(ODE, this->ode, ->Nsys())
    }

    py::object runtime() const{
        PY_GET(ODE, this->ode, ->runtime())
    }

    py::object diverges() const{
        PY_GET(ODE, this->ode, ->diverges())
    }

    py::object is_dead() const{
        PY_GET(ODE, this->ode, ->is_dead())
    }

    void reset() const{
        PY_DO(ODE, this->ode, reset())
    }

    void clear() const{
        PY_DO(ODE, this->ode, clear())
    }

    template<typename T>
    void _init_ode(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method);

    template<typename T>
    PyOdeResult _py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    template<typename T>
    PyOdeSolution _py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints);

    template<typename T>
    PyOdeResult _py_go_to(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    template<typename T>
    py::object _t_array() const;

    template<typename T>
    py::object _q_array() const;

    template<typename T>
    py::tuple _event_data(const py::str& event) const;

    template<typename T>
    py::object _solver_copy() const;

    template<typename T>
    py::dict _event_map() const{
        EventMap result = reinterpret_cast<const ODE<T>*>(this->ode)->event_map();
        return to_PyDict(result);
    }

    void* ode = nullptr; //cast to ODE<T, 0>*
    PyStruct data;
};



class PyVarODE : public PyODE{

public:

    PyVarODE(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& period, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    template<typename T>
    VariationalODE<T, 0>& varode();

    template<typename T>
    const VariationalODE<T, 0>& varode() const;

    py::object py_t_lyap() const;

    py::object py_lyap() const;

    py::object py_kicks() const;

    py::object copy() const override;

    template<typename T>
    void _init_var_ode(py::object f, const py::object& t0, const py::iterable& q0, const py::object& period, py::object jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method);

    template<typename T>
    py::object _py_t_lyap() const;

    template<typename T>
    py::object _py_lyap() const;

    template<typename T>
    py::object _py_kicks() const;

};


void py_integrate_all(const py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress);

/*
IMPLEMENTATIONS
*/




//===========================================================================================
//                          Template Method Implementations
//===========================================================================================

template<typename T>
py::object PyFuncWrapper::call_impl(const py::object& t, const py::iterable& py_q, py::args py_args) const {
    auto q = toCPP_Array<T, Array1D<T>>(py_q);
    if (static_cast<size_t>(q.size()) != Nsys || py_args.size() != Nargs){
        throw py::value_error("Invalid array sizes in ode function call");
    }
    auto args = toCPP_Array<T, std::vector<T>>(py_args);
    Array<T> res(output_shape);
    reinterpret_cast<Func<T>>(this->rhs)(res.data(), py::cast<T>(t), q.data(), args.data(), nullptr);
    return py::cast(res);
}

template<typename T>
Func<T> PyEvent::mask() const{
    if (_py_mask){
        return py_mask<T>;
    }
    else if (this->_mask != nullptr){
        return reinterpret_cast<Func<T>>(this->_mask);
    }
    else{
        return nullptr;
    }
}

template<typename T>
ObjFun<T> PyPrecEvent::when() const{
    if (_when == nullptr){
        return py_event<T>;
    }
    else{
        return reinterpret_cast<ObjFun<T>>(this->_when);
    }
}

template<typename T>
void* PyPrecEvent::get_new_event(){
    return new ObjectOwningEvent<PreciseEvent<T, 0>, PyStruct>(this->data, this->name(), this->when<T>(), _dir, this->mask<T>(), this->hide_mask(), this->_event_tol.cast<T>());
}

template<typename T>
void* PyPerEvent::get_new_event(){
    return new ObjectOwningEvent<PeriodicEvent<T, 0>, PyStruct>(this->data, this->name(), _period.cast<T>(), (_start.is_none() ? inf<T>() : _start.cast<T>()), this->mask<T>(), this->hide_mask());
}

template<typename T>
std::vector<Event<T, 0>*> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, const py::iterable& args){
    if (events.is_none()){
        return {};
    }
    std::vector<Event<T, 0>*> res;
    for (py::handle item : events){
        Event<T, 0>* ev_ptr = reinterpret_cast<Event<T, 0>*>(item.cast<PyEvent&>().toEvent(shape, args));
        res.push_back(ev_ptr);
    }
    return res;
}

template<typename T>
OdeData<T> init_ode_data(PyStruct& data, std::vector<T>& args, py::object f, const py::iterable& q0, py::object jacobian, const py::iterable& py_args, const py::iterable& events){
    std::string scalar_type = get_scalar_type<T>();
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);

    bool f_is_compiled = py::isinstance<PyFuncWrapper>(f) || py::isinstance<py::capsule>(f);
    bool jac_is_compiled = !jacobian.is_none() && (py::isinstance<PyFuncWrapper>(jacobian) || py::isinstance<py::capsule>(jacobian));
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<T> ode_rhs = {nullptr, nullptr, &data};
    if (f_is_compiled){
        if (py::isinstance<PyFuncWrapper>(f)){
            //safe approach
            auto& _f = f.cast<PyFuncWrapper&>();
            ode_rhs.rhs = reinterpret_cast<Func<T>>(_f.rhs);
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
        data.rhs = std::move(f);
        ode_rhs.rhs = py_rhs;
    }
    if (jac_is_compiled){
        if (py::isinstance<PyFuncWrapper>(jacobian)){
            //safe approach
            auto& _j = jacobian.cast<PyFuncWrapper&>();
            ode_rhs.jacobian = reinterpret_cast<Func<T>>(_j.rhs);
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
        data.jac = std::move(jacobian);
        ode_rhs.jacobian = py_jac;
    }
    for (py::handle ev : events){
        if (!py::isinstance<PyEvent>(ev)) {
            throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class, not " + py::str(ev.get_type()).cast<std::string>());
        }
        const auto& _ev = ev.cast<const PyEvent&>();
        if (_ev.scalar_type != DTYPE_MAP.at(scalar_type)){
            throw py::value_error("All event objects in 'events' must have scalar type " + scalar_type + ".");
        }
        _ev.check_sizes(_size, args.size());

    }
    return ode_rhs;
}

template<typename T>
void PySolver::init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
    std::vector<T> args;
    OdeData<T> ode_data = init_ode_data<T>(this->data, args, std::move(f), py_q0, std::move(jac), py_args, py_events);
    std::vector<Event<T, 0>*> safe_events = to_Events<T>(py_events, this->data.shape, py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i];
    }
    auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
    this->s = get_solver(name, ode_data, py::cast<T>(t0), q0, py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(first_step), dir, args, evs).release();
    for (size_t i=0; i<evs.size(); i++){
        delete safe_events[i];
    }
}

template<typename T>
py::tuple PyOdeResult::_event_data(const py::str& event) const{
    auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
    std::vector<T> t_data = r->t_filtered(event.cast<std::string>());
    Array<T> t_res(t_data.data(), t_data.size());
    Array2D<T, 0, 0> q_data = r->q_filtered(event.cast<std::string>());
    Array<T> q_res(q_data.release(), getShape(py::ssize_t(t_res.size()), this->q0_shape), true);
    return py::make_tuple(py::cast(t_res), py::cast(q_res));
}

template<typename T>
py::object PyOdeSolution::_get_frame(const py::object& t) const{
    return py::cast(Array<T>(reinterpret_cast<OdeSolution<T, 0>*>(this->res)->operator()(t.cast<T>()).data(), this->q0_shape));
}

template<typename T>
py::object PyOdeSolution::_get_array(const py::array& py_array) const{
    const auto nt = size_t(py_array.size());
    std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
    final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
    Array<T> res(final_shape);
    const auto* solution = reinterpret_cast<const OdeSolution<T, 0>*>(this->res);

    // Extract array values and cast them to T using Python's item access
    for (size_t i=0; i<nt; i++){
        py::object item = py_array.attr("flat")[py::int_(i)];
        T t_value = py::cast<T>(item);
        copy_array(res.data()+i*nsys, solution->operator()(t_value).data(), nsys);
    }
    return py::cast(res);
}

template<typename T>
void PyODE::_init_ode(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method){
    std::vector<T> args;
    OdeData<T> ode_rhs = init_ode_data<T>(data,args, std::move(f), py_q0, std::move(jacobian), py_args, events);
    std::vector<Event<T, 0>*> safe_events = to_Events<T>(events, shape(py_q0), py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i];
    }
    auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
    ode = new ODE<T, 0>(ode_rhs, py::cast<T>(t0), q0, py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(first_step), dir, args, evs, method);
    for (size_t i=0; i<evs.size(); i++){
        delete safe_events[i];
    }
}

template<typename T>
PyOdeResult PyODE::_py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    auto* ptr = new OdeResult<T, 0>(reinterpret_cast<ODE<T, 0>*>(ode)->integrate(py::cast<T>(interval), to_step_sequence<T>(t_eval), to_Options(event_options), max_prints));
    return PyOdeResult(ptr, this->data.shape, this->scalar_type);
}

template<typename T>
PyOdeSolution PyODE::_py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints){
    auto* ptr = new OdeSolution<T, 0>(reinterpret_cast<ODE<T, 0>*>(ode)->rich_integrate(py::cast<T>(interval), to_Options(event_options), max_prints));
    return PyOdeSolution(ptr, this->data.shape, this->scalar_type);
}

template<typename T>
PyOdeResult PyODE::_py_go_to(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    auto* ptr = new OdeResult<T, 0>(reinterpret_cast<ODE<T, 0>*>(ode)->go_to(py::cast<T>(t), to_step_sequence<T>(t_eval), to_Options(event_options), max_prints));
    return PyOdeResult(ptr, this->data.shape, this->scalar_type);
}

template<typename T>
py::object PyODE::_t_array() const{
    auto* r = reinterpret_cast<ODE<T>*>(this->ode);
    return py::cast(NdView<const T>(r->t().data(), r->t().size()));
}

template<typename T>
py::object PyODE::_q_array() const{
    auto* r = reinterpret_cast<ODE<T>*>(this->ode);
    return py::cast(NdView<const T>(r->q().data(), getShape(py::ssize_t(r->t().size()), this->data.shape)));
}

template<typename T>
py::tuple PyODE::_event_data(const py::str& event) const{
    std::vector<T> t_data = reinterpret_cast<const ODE<T, 0>*>(ode)->t_filtered(event.cast<std::string>());
    Array2D<T, 0, 0> q_data = reinterpret_cast<const ODE<T, 0>*>(ode)->q_filtered(event.cast<std::string>());
    Array<T> q_res(q_data.release(), getShape(py::ssize_t(t_data.size()), data.shape), true);
    return py::make_tuple(py::cast(Array<T>(t_data.data(), t_data.size())), py::cast(q_res));
}

template<typename T>
py::object PyODE::_solver_copy() const{
    auto* ode_ptr = reinterpret_cast<const ODE<T, 0>*>(ode);
    auto* solver_clone = ode_ptr->solver()->clone();
    if (ode_ptr->solver()->name() == "RK45"){
        return py::cast(PyRK45(solver_clone, data, this->scalar_type));
    }
    else if (ode_ptr->solver()->name() == "DOP853"){
        return py::cast(PyDOP853(solver_clone, data, this->scalar_type));
    }
    else if (ode_ptr->solver()->name() == "RK23"){
        return py::cast(PyRK23(solver_clone, data, this->scalar_type));
    }
    else if (ode_ptr->solver()->name() == "BDF"){
        return py::cast(PyBDF(solver_clone, data, this->scalar_type));
    }
    else{
        throw py::value_error("Unregistered solver!");
    }
}

template<typename T>
void PyVarODE::_init_var_ode(py::object f, const py::object& t0, const py::iterable& q0, const py::object& period, py::object jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method){
    std::vector<T> args;
    OdeData<T> ode_rhs = init_ode_data<T>(this->data, args, std::move(f), q0, std::move(jac), py_args, events);
    Array1D<T> q0_ = toCPP_Array<T, Array1D<T>>(q0);
    std::vector<Event<T, 0>*> safe_events = to_Events<T>(events, shape(q0), py_args);
    std::vector<const Event<T, 0>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i];
    }
    this->ode = new VariationalODE<T, 0>(ode_rhs, py::cast<T>(t0), q0_, py::cast<T>(period), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(first_step), dir, args, evs, method.cast<std::string>());
    for (size_t i=0; i<evs.size(); i++){
        delete safe_events[i];
    }
}

template<typename T>
VariationalODE<T, 0>& PyVarODE::varode(){
    return *static_cast<VariationalODE<T, 0>*>(this->ode);
}

template<typename T>
const VariationalODE<T, 0>& PyVarODE::varode() const {
    return *static_cast<const VariationalODE<T, 0>*>(this->ode);
}

template<typename T>
py::object PyVarODE::_py_t_lyap() const{
    const auto& vode = varode<T>();
    NdView<const T> res(vode.t_lyap().data(), vode.t_lyap().size());
    return py::cast(res);
}

template<typename T>
py::object PyVarODE::_py_lyap() const{
    const auto& vode = varode<T>();
    NdView<const T> res(vode.lyap().data(), vode.t_lyap().size());
    return py::cast(res);
}

template<typename T>
py::object PyVarODE::_py_kicks() const{
    const auto& vode = varode<T>();
    NdView<const T> res(vode.kicks().data(), vode.t_lyap().size());
    return py::cast(res);
}

#endif
