#include "pyode.hpp"
//===========================================================================================
//                                      PyFuncWrapper
//===========================================================================================

PyFuncWrapper::PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type) : DtypeDispatcher(scalar_type), rhs(open_capsule<void*>(obj)), Nsys(static_cast<size_t>(Nsys)), output_shape(static_cast<size_t>(output_shape.size())), Nargs(static_cast<size_t>(Nargs)) {
    copy_array(this->output_shape.data(), output_shape.data(), this->output_shape.size());
    long s = 1;
    for (long i : this->output_shape){
        s *= i;
    }
    this->output_size = size_t(s);
}

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

py::object PyFuncWrapper::call(const py::object& t, const py::iterable& py_q, const py::args& py_args) const{
    EXECUTE(return, this->call_impl, (t, py_q, py_args);, return py::none();)
}

//===========================================================================================
//                                      PyEvent
//===========================================================================================

PyEvent::PyEvent(std::string name, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs) : DtypeDispatcher(scalar_type), _name(std::move(name)), _hide_mask(hide_mask), _Nsys(Nsys), _Nargs(Nargs){
    if (py::isinstance<py::capsule>(mask)){
        this->_mask = open_capsule<void*>(mask);
    }
    else if (py::isinstance<py::function>(mask)){
        data.mask = std::move(mask);
        this->_py_mask = true;
    }
}

py::str PyEvent::name() const{
    return _name;
}

py::bool_ PyEvent::hide_mask() const {
    return _hide_mask;
}

bool PyEvent::is_lowlevel() const{
    return data.event.is_none() && data.mask.is_none();
}

void PyEvent::check_sizes(size_t Nsys, size_t Nargs) const{
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

PyEvent::~PyEvent(){}

//===========================================================================================
//                                      PyPrecEvent
//===========================================================================================

PyPrecEvent::PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, py::object event_tol, const std::string& scalar_type, size_t Nsys, size_t Nargs) : PyEvent(std::move(name), std::move(mask), hide_mask, scalar_type, Nsys, Nargs), _dir(sgn(dir)), _event_tol(std::move(event_tol)){
    if (py::isinstance<py::capsule>(when)){
        this->_when = open_capsule<void*>(when);
    }
    else if (py::isinstance<py::function>(when)){
        this->data.event = std::move(when);
    }

}

py::object PyPrecEvent::event_tol() const {
    return _event_tol;
}

void* PyPrecEvent::toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) {
    if (this->is_lowlevel()){
        for (const py::handle& arg : args){
            if (PyNumber_Check(arg.ptr()) == 0){
                throw py::value_error("All args must be numbers");
            }
        }
    }
    this->data.py_args = args;
    this->data.shape = shape;
    EXECUTE(return, this->get_new_event, ();, return nullptr;)
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

//===========================================================================================
//                                      PyPerEvent
//===========================================================================================

PyPerEvent::PyPerEvent(std::string name, py::object period, py::object start, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs):PyEvent(std::move(name), std::move(mask), hide_mask, scalar_type, Nsys, Nargs), _period(std::move(period)), _start(std::move(start)){}

void* PyPerEvent::toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) {

    if (this->is_lowlevel()){
        for (const py::handle& arg : args){
            if (PyNumber_Check(arg.ptr())==0){
                throw py::value_error("All args must be numbers");
            }
        }
    }
    this->data.py_args = args;
    this->data.shape = shape;
    EXECUTE(return, this->get_new_event, ();, return nullptr;)
}

template<typename T>
void* PyPerEvent::get_new_event(){
    return new ObjectOwningEvent<PeriodicEvent<T, 0>, PyStruct>(this->data, this->name(), _period.cast<T>(), (_start.is_none() ? inf<T>() : _start.cast<T>()), this->mask<T>(), this->hide_mask());
}

py::object PyPerEvent::period() const{
    return _period;
}

py::object PyPerEvent::start() const{
    return _start;
}

//===========================================================================================
//                                      Helper Functions
//===========================================================================================

inline std::vector<EventOptions> to_Options(const py::iterable& d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        auto opt = py::cast<EventOptions>(item);
        result.emplace_back(opt);
    }
    result.shrink_to_fit();
    return result;
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

// init_ode_data
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

//===========================================================================================
//                                      PySolver
//===========================================================================================

PySolver::PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type) : DtypeDispatcher(scalar_type){
    EXECUTE(, this->init_solver, (f, jac, t0, py_q0, rtol, atol, min_step, max_step, first_step, dir, py_args, py_events, name);, )
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

PySolver::PySolver(void* solver, PyStruct py_data, int scalar_type) : DtypeDispatcher(scalar_type), data(std::move(py_data)){
    EXECUTE(, reinterpret_cast<OdeSolver, *>(solver)->set_obj(&data);, )
    this->s = solver;
}

PySolver::PySolver(const PySolver& other) : DtypeDispatcher(other), data(other.data) {
    EXECUTE(this->s =, reinterpret_cast<OdeSolver, *>(other.s)->clone();, )
    EXECUTE(, reinterpret_cast<OdeSolver, *>(this->s)->set_obj(&data);, )
}

PySolver::PySolver(PySolver&& other) noexcept : DtypeDispatcher(std::move(other)), s(other.s), data(std::move(other.data)) {
    other.s = nullptr;
    EXECUTE(, reinterpret_cast<OdeSolver, *>(this->s)->set_obj(&data);, )
}


PySolver& PySolver::operator=(const PySolver& other){
    if (&other != this){
        data = other.data;
        EXECUTE(delete, reinterpret_cast<OdeSolver, *>(this->s);, )
        EXECUTE(this->s =, reinterpret_cast<OdeSolver, *>(other.s)->clone();, )
        EXECUTE(, reinterpret_cast<OdeSolver, *>(this->s)->set_obj(&data);, )
    }
    return *this;
}


PySolver& PySolver::operator=(PySolver&& other) noexcept{
    if (&other != this){
        data = std::move(other.data);
        EXECUTE(delete, reinterpret_cast<OdeSolver, *>(this->s);, )
        this->s = other.s;
        EXECUTE(, reinterpret_cast<OdeSolver, *>(this->s)->set_obj(&data);, )
        other.s = nullptr;
    }
    return *this;
}


PySolver::~PySolver(){
    EXECUTE(delete, reinterpret_cast<OdeSolver, *>(this->s);, )
}

//===========================================================================================
//                                      PyRK23
//===========================================================================================

PyRK23::PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK23", scalar_type){
}

py::object PyRK23::copy() const{
    return py::cast(PyRK23(*this));
}

//===========================================================================================
//                                      PyRK45
//===========================================================================================

PyRK45::PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "RK45", scalar_type){}

py::object PyRK45::copy() const{
    return py::cast(PyRK45(*this));
}

//===========================================================================================
//                                      PyDOP853
//===========================================================================================

PyDOP853::PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "DOP853", scalar_type){}

py::object PyDOP853::copy() const{
    return py::cast(PyDOP853(*this));
}

//===========================================================================================
//                                      PyBDF
//===========================================================================================

PyBDF::PyBDF(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(f, jac, t0, q0, rtol, atol, min_step, max_step, first_step, dir, args, events, "BDF", scalar_type){}

py::object PyBDF::copy() const{
    return py::cast(PyBDF(*this));
}

//===========================================================================================
//                                      PyOdeResult
//===========================================================================================



PyOdeResult::PyOdeResult(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type): DtypeDispatcher(scalar_type), res(result), q0_shape(q0_shape){}


PyOdeResult::PyOdeResult(const PyOdeResult& other) : DtypeDispatcher(other.scalar_type), q0_shape(other.q0_shape) {
    EXECUTE(this->res =, reinterpret_cast<OdeResult, *>(other.res)->clone();,)
}


PyOdeResult::PyOdeResult(PyOdeResult&& other) noexcept : DtypeDispatcher(other.scalar_type), res(other.res), q0_shape(std::move(other.q0_shape)) {
    other.res = nullptr;
}


PyOdeResult::~PyOdeResult(){
    EXECUTE(delete, reinterpret_cast<OdeResult, *>(this->res);, )
    res = nullptr;
}

PyOdeResult& PyOdeResult::operator=(const PyOdeResult& other){
    if (&other != this){
        EXECUTE(delete, reinterpret_cast<OdeResult, *>(this->res);, )
        EXECUTE(this->res =, reinterpret_cast<OdeResult, *>(other.res)->clone();,)
        q0_shape = other.q0_shape;
    }
    return *this;
}


PyOdeResult& PyOdeResult::operator=(PyOdeResult&& other) noexcept{
    if (&other != this){
        EXECUTE(delete, reinterpret_cast<OdeResult, *>(this->res);, )
        this->res = other.res;
        q0_shape = other.q0_shape;
        other.res = nullptr;
    }
    return *this;
}

template<typename T>
py::tuple PyOdeResult::_event_data(const py::str& event) const{
    auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
    std::vector<T> t_data = r->t_filtered(event.cast<std::string>());
    Array<T> t_res(t_data.data());
    Array2D<T> q_data = r->q_filtered(event.cast<std::string>());
    return py::make_tuple(py::cast(t_res), py::cast(Array<T>(q_data.release(), getShape(t_res.size(), this->q0_shape), true)));
}


//===========================================================================================
//                                      PyOdeSolution
//===========================================================================================


PyOdeSolution::PyOdeSolution(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type) : PyOdeResult(result, q0_shape, scalar_type), nsys(prod(q0_shape)) {}

py::object PyOdeSolution::operator()(const py::object& t) const{
    try {
        // Try to convert t to a numpy array
        py::array arr = py::array::ensure(t);
        EXECUTE(return, this->_get_array, (arr);, return py::none();)
    } catch (const py::cast_error&) {
        // If conversion fails, treat as a scalar
        EXECUTE(return, this->_get_frame, (t);, return py::none();)
    }
}

//===========================================================================================
//                                      PyODE
//===========================================================================================


PyODE::PyODE(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type) : DtypeDispatcher(scalar_type){
    EXECUTE(, this->_init_ode, (f, t0, py_q0, jacobian, rtol, atol, min_step, max_step, first_step, dir, py_args, events, method);, )
}

PyODE::PyODE(const PyODE& other) : DtypeDispatcher(other.scalar_type), data(other.data){
    EXECUTE(this->ode =, reinterpret_cast<ODE, *>(other.ode)->clone();, )
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
}

PyODE::PyODE(PyODE&& other) noexcept : DtypeDispatcher(std::move(other)), ode(other.ode), data(std::move(other.data)){
    other.ode = nullptr;
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
}

PyODE& PyODE::operator=(const PyODE& other){
    if (&other == this){
        return *this;
    }
    EXECUTE(delete, reinterpret_cast<ODE, *>(this->ode);, )
    EXECUTE(this->ode =, reinterpret_cast<ODE, *>(other.ode)->clone();, )
    data = other.data;
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
    return *this;
}

PyODE& PyODE::operator=(PyODE&& other) noexcept {
    if (&other == this){
        return *this;
    }
    EXECUTE(delete, reinterpret_cast<ODE, *>(this->ode);, )
    this->ode = other.ode;
    other.ode = nullptr;
    data = std::move(other.data);
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
    return *this;
}

PyODE::~PyODE(){
    EXECUTE(delete, reinterpret_cast<ODE, *>(this->ode);, )
}

py::object PyODE::py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    PY_GET_TEMPLATE(this->_py_integrate, (interval, t_eval, event_options, max_prints))
}

py::object PyODE::py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints){
    PY_GET_TEMPLATE(this->_py_rich_integrate, (interval, event_options, max_prints))}

py::object PyODE::py_go_to(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){
    PY_GET_TEMPLATE(this->_py_go_to, (t, t_eval, event_options, max_prints))}

py::object PyODE::t_array() const{
    EXECUTE(return, this->_t_array, ();, return py::none();)
}

py::object PyODE::q_array() const{
    EXECUTE(return, this->_q_array, ();, return py::none();)
}

py::tuple PyODE::event_data(const py::str& event) const{
    EXECUTE(return, this->_event_data, (event);, return py::none();)
}

py::object PyODE::copy() const{
    return py::cast(PyODE(*this));
}

py::object PyODE::solver_copy() const{
    EXECUTE(return, this->_solver_copy, ();, return py::none();)
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

//===========================================================================================
//                                      PyVarODE
//===========================================================================================


PyVarODE::PyVarODE(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& period, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type):PyODE(scalar_type){
    EXECUTE(, this->_init_var_ode, (f, t0, q0, period, jac, rtol, atol, min_step, max_step, first_step, dir, py_args, events, method);, )
}

py::object PyVarODE::py_t_lyap() const{
    EXECUTE(return, this->_py_t_lyap, ();, return py::none();)
}

py::object PyVarODE::py_lyap() const{
    EXECUTE(return, this->_py_lyap, ();, return py::none();)
}

py::object PyVarODE::py_kicks() const{
    EXECUTE(return, this->_py_kicks, ();, return py::none();)
}

py::object PyVarODE::copy() const{
    return py::cast(PyVarODE(*this));
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



void py_integrate_all(const py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress){
    // Separate lists for each numeric type
    std::vector<ODE<double, 0>*> array_double;
    std::vector<ODE<float, 0>*> array_float;
    std::vector<ODE<long double, 0>*> array_longdouble;
    std::vector<ODE<mpfr::mpreal, 0>*> array_mpreal;

    // Iterate through the list and identify each PyODE type
    for (const py::handle& item : list) {
        try {
            auto& pyode = item.cast<PyODE&>();
            
            // Use the scalar_type to determine which array to add to
            switch (pyode.scalar_type) {
                case 0:
                    array_double.push_back(reinterpret_cast<ODE<double>*>(pyode.ode));
                    break;
                case 1:
                    array_longdouble.push_back(reinterpret_cast<ODE<long double>*>(pyode.ode));
                    break;
                case 2:
                    array_mpreal.push_back(reinterpret_cast<ODE<mpfr::mpreal>*>(pyode.ode));
                    break;
                case 3:
                    array_float.push_back(reinterpret_cast<ODE<float>*>(pyode.ode));
                    break;
                default:
                    throw py::value_error("Unregistered scalar_type in PyODE object.");
            }
        } catch (const py::cast_error&) {
            // If cast failed, throw an error
            throw py::value_error("List item is not a recognized PyODE object type.");
        }
    }

    // Convert event_options once (it's not templated)
    auto options = to_Options(event_options);

    // Call integrate_all for each type group that has elements
    if (!array_double.empty()) {
        integrate_all<double, 0>(array_double, interval, to_step_sequence<double>(t_eval), options, threads, display_progress);
    }
    if (!array_float.empty()) {
        integrate_all<float, 0>(array_float, static_cast<float>(interval), to_step_sequence<float>(t_eval), options, threads, display_progress);
    }
    if (!array_longdouble.empty()) {
        integrate_all<long double, 0>(array_longdouble, static_cast<long double>(interval), to_step_sequence<long double>(t_eval), options, threads, display_progress);
    }

    if (!array_mpreal.empty()) {
        integrate_all<mpfr::mpreal, 0>(array_mpreal, mpfr::mpreal(interval), to_step_sequence<mpfr::mpreal>(t_eval), options, threads, display_progress);
    }
}

PYBIND11_MODULE(odesolvers, m) {

    py::class_<PyFuncWrapper>(m, "LowLevelFunction")
        .def(py::init<py::capsule, py::ssize_t, py::array_t<py::ssize_t>, py::ssize_t, py::str>(),
            py::arg("pointer"),
            py::arg("input_size"),
            py::arg("output_shape"),
            py::arg("Nargs"),
            py::arg("scalar_type")="double")
        .def("__call__", &PyFuncWrapper::call, py::arg("t"), py::arg("q"))
        .def_property_readonly("scalar_type", [](const PyFuncWrapper& self){return SCALAR_TYPE[self.scalar_type];});

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


    py::class_<PyEvent>(m, "Event")
        .def_property_readonly("name", [](const PyEvent& self){
            return self.name();
        })
        .def_property_readonly("hides_mask", [](const PyEvent& self){
            return self.hide_mask();
        })
        .def_property_readonly("scalar_type", [](const PyEvent& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyPrecEvent, PyEvent>(m, "PreciseEvent")
        .def(py::init<std::string, py::object, int, py::object, bool, py::object, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("dir")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("event_tol", [](const PyPrecEvent& self){
            return self.event_tol();
        });

    py::class_<PyPerEvent, PyEvent>(m, "PeriodicEvent")
        .def(py::init<std::string, py::object, py::object, py::object, bool, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("start")=py::none(),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("period", [](const PyPerEvent& self){
                return self.period();})
        .def_property_readonly("start", [](const PyPerEvent& self){
            return self.start();});


    py::class_<PySolver>(m, "OdeSolver")
        .def_property_readonly("t", [](const PySolver& self){return self.t();})
        .def_property_readonly("q", [](const PySolver& self){return self.q();})
        .def_property_readonly("stepsize", [](const PySolver& self){return self.stepsize();})
        .def_property_readonly("diverges", [](const PySolver& self){return self.diverges();})
        .def_property_readonly("is_dead", [](const PySolver& self){return self.is_dead();})
        .def_property_readonly("Nsys", [](const PySolver& self){return self.Nsys();})
        .def("show_state", [](const PySolver& self, int digits){
                self.show_state(digits);
            },
            py::arg("digits") = 8
        )
        .def("advance", [](const PySolver& self){self.advance();})
        .def("advance_to_event", [](const PySolver& self){self.advance_to_event();})
        .def("reset", [](const PySolver& self){self.reset();})
        .def("copy", [](const PySolver& self){return self.copy();})
        .def_property_readonly("scalar_type", [](const PySolver& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyRK23, PySolver>(m, "RK23")
        .def(py::init<PyRK23>(), py::arg("result"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyRK45, PySolver>(m, "RK45")
        .def(py::init<PyRK45>(), py::arg("result"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyDOP853, PySolver>(m, "DOP853")
        .def(py::init<PyDOP853>(), py::arg("result"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyBDF, PySolver>(m, "BDF")
        .def(py::init<PyBDF>(), py::arg("result"))
        .def(py::init<py::object, py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("jac"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyOdeResult>(m, "OdeResult")
        .def(py::init<PyOdeResult>(), py::arg("result"))
        .def_property_readonly("t", [](const PyOdeResult& self){
            return self.t();
        })
        .def_property_readonly("q", [](const PyOdeResult& self){
            return self.q();
        })
        .def_property_readonly("event_map", [](const PyOdeResult& self){
            return self.event_map();
        })
        .def("event_data", &PyOdeResult::event_data, py::arg("event"))
        .def_property_readonly("diverges", [](const PyOdeResult& self){return self.diverges();})
        .def_property_readonly("success", [](const PyOdeResult& self){return self.success();})
        .def_property_readonly("runtime", [](const PyOdeResult& self){return self.runtime();})
        .def_property_readonly("message", [](const PyOdeResult& self){return self.message();})
        .def("examine", [](const PyOdeResult& self){self.examine();})
        .def_property_readonly("scalar_type", [](const PyOdeResult& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyOdeSolution, PyOdeResult>(m, "OdeSolution")
        .def(py::init<PyOdeSolution>(), py::arg("result"))
        .def("__call__", [](const PyOdeSolution& self, const py::object& t){return self(t);});

    py::class_<PyODE>(m, "LowLevelODE")
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyODE>(), py::arg("result"))
        .def("solver", [](const PyODE& self){return self.solver_copy();})
        .def("integrate", &PyODE::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("go_to", &PyODE::py_go_to,
            py::arg("t"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("rich_integrate", &PyODE::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("copy", [](const PyODE& self){return self.copy();})
        .def("reset", [](const PyODE& self){self.reset();})
        .def("clear", [](const PyODE& self){self.reset();})
        .def_property_readonly("t", [](const PyODE& self){return self.t_array();})
        .def_property_readonly("q", [](const PyODE& self){return self.q_array();})
        .def("event_data", [](const PyODE& self, const py::str& event){
            return self.event_data(event);
        }, py::arg("event"))
        .def_property_readonly("event_map", [](const PyODE& self){
            return self.event_map();
        })
        .def_property_readonly("Nsys", [](const PyODE& self){
            return self.Nsys();
        })
        .def_property_readonly("runtime", [](const PyODE& self){
            return self.runtime();
        })
        .def_property_readonly("diverges", [](const PyODE& self){
            return self.diverges();
        })
        .def_property_readonly("is_dead", [](const PyODE& self){
            return self.is_dead();
        })
        .def_property_readonly("scalar_type", [](const PyODE& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyVarODE, PyODE>(m, "VariationalLowLevelODE")
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("first_step")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyVarODE>(), py::arg("result"))
        .def_property_readonly("t_lyap", &PyVarODE::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE::py_lyap)
        .def_property_readonly("kicks", &PyVarODE::py_kicks)
        .def("copy", [](const PyVarODE& self){return self.copy();});


    
    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

    m.def("set_mpreal_prec",
      [](int bits) {
          mpfr::mpreal::set_default_prec(bits);
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
          return mpfr::mpreal::get_default_prec();
      });
}