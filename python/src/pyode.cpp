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

//===========================================================================================
//                                      PySolver
//===========================================================================================

PySolver::PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& first_step, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type) : DtypeDispatcher(scalar_type){
    EXECUTE(, this->init_solver, (f, jac, t0, py_q0, rtol, atol, min_step, max_step, first_step, dir, py_args, py_events, name);, )
}

PySolver::PySolver(void* solver, PyStruct py_data, int scalar_type) : DtypeDispatcher(scalar_type), data(std::move(py_data)){
    EXECUTE(, reinterpret_cast<OdeRichSolver, *>(solver)->set_obj(&data);, )
    this->s = solver;
}

PySolver::PySolver(const PySolver& other) : DtypeDispatcher(other), data(other.data) {
    EXECUTE(this->s =, reinterpret_cast<OdeRichSolver, *>(other.s)->clone();, )
    EXECUTE(, reinterpret_cast<OdeRichSolver, *>(this->s)->set_obj(&data);, )
}

PySolver::PySolver(PySolver&& other) noexcept : DtypeDispatcher(std::move(other)), s(other.s), data(std::move(other.data)) {
    other.s = nullptr;
    EXECUTE(, reinterpret_cast<OdeRichSolver, *>(this->s)->set_obj(&data);, )
}


PySolver& PySolver::operator=(const PySolver& other){
    if (&other != this){
        data = other.data;
        EXECUTE(delete, reinterpret_cast<OdeRichSolver, *>(this->s);, )
        EXECUTE(this->s =, reinterpret_cast<OdeRichSolver, *>(other.s)->clone();, )
        EXECUTE(, reinterpret_cast<OdeRichSolver, *>(this->s)->set_obj(&data);, )
    }
    return *this;
}


PySolver& PySolver::operator=(PySolver&& other) noexcept{
    if (&other != this){
        data = std::move(other.data);
        EXECUTE(delete, reinterpret_cast<OdeRichSolver, *>(this->s);, )
        this->s = other.s;
        EXECUTE(, reinterpret_cast<OdeRichSolver, *>(this->s)->set_obj(&data);, )
        other.s = nullptr;
    }
    return *this;
}


PySolver::~PySolver(){
    EXECUTE(delete, reinterpret_cast<OdeRichSolver, *>(this->s);, )
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

PyODE::PyODE(const PyODE& other) : DtypeDispatcher(other.scalar_type), data(other.data), is_lowlevel(other.is_lowlevel){
    EXECUTE(this->ode =, reinterpret_cast<ODE, *>(other.ode)->clone();, )
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
}

PyODE::PyODE(PyODE&& other) noexcept : DtypeDispatcher(std::move(other)), ode(other.ode), data(std::move(other.data)), is_lowlevel(other.is_lowlevel){
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
    is_lowlevel = other.is_lowlevel;
    EXECUTE(, reinterpret_cast<ODE, *>(this->ode)->set_obj(&data);, )
    return *this;
}

PyODE& PyODE::operator=(PyODE&& other) noexcept {
    if (&other == this){
        return *this;
    }
    EXECUTE(delete, reinterpret_cast<ODE, *>(this->ode);, )
    this->is_lowlevel = other.is_lowlevel;
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


//===========================================================================================
//                                      Additional functions
//===========================================================================================

bool all_are_lowlevel(const py::iterable& events){
    if (events.is_none()){
        return true;
    }
    for (py::handle item : events){
        if (!item.cast<PyEvent&>().is_lowlevel()){
            return false;
        }
    }
    return true;
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
            if (!pyode.is_lowlevel) {
                throw py::value_error("All ODE's in integrate_all must use only compiled functions, and no pure python functions");
            }
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


//===========================================================================================
//                                      PYTHON EXTENSION
//===========================================================================================

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
        .def_property_readonly("name", &PyEvent::name)
        .def_property_readonly("hides_mask", &PyEvent::hide_mask)
        .def_property_readonly("scalar_type", [](const PyEvent& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyPrecEvent, PyEvent>(m, "PreciseEvent")
        .def(py::init<std::string, py::object, int, py::object, bool, py::object, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("direction")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("event_tol", &PyPrecEvent::event_tol);

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
        .def_property_readonly("period", &PyPerEvent::period)
        .def_property_readonly("start", &PyPerEvent::start);


    py::class_<PySolver>(m, "OdeRichSolver")
        .def_property_readonly("t", &PySolver::t)
        .def_property_readonly("q", &PySolver::q)
        .def_property_readonly("stepsize", &PySolver::stepsize)
        .def_property_readonly("diverges", &PySolver::diverges)
        .def_property_readonly("is_dead", &PySolver::is_dead)
        .def_property_readonly("Nsys", &PySolver::Nsys)
        .def("show_state", &PySolver::show_state,
            py::arg("digits") = 8
        )
        .def("advance", &PySolver::advance)
        .def("advance_to_event", &PySolver::advance_to_event)
        .def("reset", &PySolver::reset)
        .def("copy", &PySolver::copy)
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
        .def_property_readonly("t", &PyOdeResult::t)
        .def_property_readonly("q", &PyOdeResult::q)
        .def_property_readonly("event_map", &PyOdeResult::event_map)
        .def("event_data", &PyOdeResult::event_data, py::arg("event"))
        .def_property_readonly("diverges", &PyOdeResult::diverges)
        .def_property_readonly("success", &PyOdeResult::success)
        .def_property_readonly("runtime", &PyOdeResult::runtime)
        .def_property_readonly("message", &PyOdeResult::message)
        .def("examine", &PyOdeResult::examine)
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
        .def("solver", &PyODE::solver_copy)
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
        .def("copy", &PyODE::copy)
        .def("reset", &PyODE::reset)
        .def("clear", &PyODE::clear)
        .def("event_data", &PyODE::event_data, py::arg("event"))
        .def_property_readonly("t", &PyODE::t_array)
        .def_property_readonly("q", &PyODE::q_array)
        .def_property_readonly("event_map", &PyODE::event_map)
        .def_property_readonly("Nsys", &PyODE::Nsys)
        .def_property_readonly("runtime", &PyODE::runtime)
        .def_property_readonly("diverges", &PyODE::diverges)
        .def_property_readonly("is_dead", &PyODE::is_dead)
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
        .def("copy", &PyVarODE::copy);



    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

    m.def("set_mpreal_prec",
      &mpfr::mpreal::set_default_prec,
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", &mpfr::mpreal::get_default_prec);
}
