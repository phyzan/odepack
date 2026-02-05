#include "pyode.hpp"

namespace ode{

//===========================================================================================
//                                      DtypeDispatcher
//===========================================================================================


DtypeDispatcher::DtypeDispatcher(const std::string& dtype_){
    this->scalar_type = DTYPE_MAP.at(dtype_);
}

DtypeDispatcher::DtypeDispatcher(int dtype_) : scalar_type(dtype_) {}

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

    return DISPATCH(py::object,
        auto q = toCPP_Array<T, Array1D<T>>(py_q);
        if (static_cast<size_t>(q.size()) != Nsys || py_args.size() != Nargs){
            throw py::value_error("Invalid array sizes in ode function call");
        }
        auto args = toCPP_Array<T, std::vector<T>>(py_args);
        Array<T> res(output_shape.data(), output_shape.size());
        reinterpret_cast<Func<T>>(this->rhs)(res.data(), py::cast<T>(t), q.data(), args.data(), nullptr);
        return py::cast(res);
    )
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
    data.is_lowlevel = data.mask.is_none() && data.event.is_none();
}

py::str PyEvent::name() const{
    return _name;
}

py::bool_ PyEvent::hide_mask() const {
    return _hide_mask;
}

bool PyEvent::is_lowlevel() const{
    return data.is_lowlevel;
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

    return DISPATCH(void*,
        return this->get_new_event<T>();
    )
}

//===========================================================================================
//                                      PyPerEvent
//===========================================================================================

PyPerEvent::PyPerEvent(std::string name, py::object period, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs):PyEvent(std::move(name), std::move(mask), hide_mask, scalar_type, Nsys, Nargs), _period(std::move(period)) {}

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
    return DISPATCH(void*,
        return this->get_new_event<T>();
    )
}

py::object PyPerEvent::period() const{
    return _period;
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

PySolver::PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type) : DtypeDispatcher(scalar_type){

    DISPATCH(void,
        this->init_solver<T>(f, jac, t0, py_q0, rtol, atol, min_step, max_step, stepsize, dir, py_args, py_events, name);
    )
}

PySolver::PySolver(void* solver, PyStruct py_data, int scalar_type) : DtypeDispatcher(scalar_type), data(std::move(py_data)){
    this->s = solver;
    DISPATCH(void, cast<T>()->set_obj(&data);)
}

PySolver::PySolver(const PySolver& other) : DtypeDispatcher(other), data(other.data){
    DISPATCH(void, this->s = other.template cast<T>()->clone();)
    DISPATCH(void, cast<T>()->set_obj(&data);)
}

PySolver::PySolver(PySolver&& other) noexcept : DtypeDispatcher(std::move(other)), s(other.s), data(std::move(other.data)){
    other.s = nullptr;
    DISPATCH(void, cast<T>()->set_obj(&data);)
}


PySolver& PySolver::operator=(const PySolver& other){
    if (&other != this){
        data = other.data;
        DISPATCH(void, delete cast<T>();)
        DISPATCH(void, this->s = other.template cast<T>()->clone();)
        DISPATCH(void, cast<T>()->set_obj(&data);)
    }
    return *this;
}


PySolver& PySolver::operator=(PySolver&& other) noexcept{
    if (&other != this){
        data = std::move(other.data);
        DISPATCH(void, delete cast<T>();)
        this->s = other.s;
        DISPATCH(void, cast<T>()->set_obj(&data);)
        other.s = nullptr;
    }
    return *this;
}


PySolver::~PySolver(){
    DISPATCH(void, delete cast<T>();)
}


py::object PySolver::t() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->t());
    )
}

py::object PySolver::t_old() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->t_old());
    )
}

py::object PySolver::q() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->vector());
    )
}

py::object PySolver::q_old() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->vector_old());
    )
}

py::object PySolver::stepsize() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->stepsize());
    )
}

py::object PySolver::diverges() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->diverges());
    )
}

py::object PySolver::is_dead() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->is_dead());
    )
}

py::object PySolver::Nsys() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->Nsys());
    )
}

py::object PySolver::n_evals_rhs() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->n_evals_rhs());
    )
}

void PySolver::show_state(int digits) const{
    DISPATCH(void,
        return cast<T>()->show_state(digits);
    )
}

py::object PySolver::advance() {
    return DISPATCH(py::object,
        return py::cast(cast<T>()->advance());
    )
}

py::object PySolver::advance_to_event() {
    return DISPATCH(py::object,
        return py::cast(cast<T>()->advance_to_event());
    )
}

py::object PySolver::advance_until(const py::object& time) {
    return DISPATCH(py::object,
        return py::cast(cast<T>()->advance_until(time.cast<T>()));
    )
}

void PySolver::reset() {
    DISPATCH(void,
        return cast<T>()->reset();
    )
}

bool PySolver::set_ics(const py::object& t0, const py::iterable& py_q0, const py::object& dt, int direction) {
    if (direction != 1 && direction != -1 && direction != 0){
        throw py::value_error("Direction must be either +1 or -1 or 0 (default)");
    }
    return DISPATCH(bool,
        if (dt.cast<T>() < 0){
            throw py::value_error("Stepsize cannot be negative");
        }
        auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
        if (size_t(q0.size()) != cast<T>()->Nsys()){
            throw py::value_error("Invalid size of initial condition array");
        }
        return cast<T>()->set_ics(t0.cast<T>(), q0.data(), dt.cast<T>(), direction);
    )
}

bool PySolver::resume() {
    return DISPATCH(bool,
        return cast<T>()->resume();
    )
}

py::str PySolver::message() const{
    return DISPATCH(py::str,
        return py::cast(cast<T>()->message());
    )
}

py::object PySolver::py_at_event() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->at_event());
    )
}

py::object PySolver::py_event_located(const py::str& name) const{
    return DISPATCH(py::object,
        EventView<T> events = cast<T>()->current_events();
        std::string ev = name.cast<std::string>();
        for (size_t i=0; i<events.size(); i++){
            if (events[i]->name() == ev){
                return py::cast(true);
            }
        }
        return py::cast(false);
    )

}


//===========================================================================================
//                                      PyVarSolver
//===========================================================================================


PyVarSolver::PyVarSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const std::string& method, const std::string& scalar_type) : PySolver(scalar_type) {
    DISPATCH(void,
        std::vector<T> args;
        OdeData<Func<T>, void> ode_data = init_ode_data<T>( this->data, args, f, py_q0, jac, py_args, py::list());
        auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
        if ((q0.size() & 1) != 0){
            throw py::value_error("Variational solvers require an even number of system size");
        }

        this->s = VariationalSolver<T, 0, Func<T>, void>(ode_data, t0.cast<T>(), q0.data(), q0.size() / 2, period.cast<T>(), rtol.cast<T>(), atol.cast<T>(), min_step.cast<T>(), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), stepsize.cast<T>(), dir, args, method).release();

    )
}

PyVarSolver::PyVarSolver(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

inline py::object PyVarSolver::py_logksi() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().logksi());
    )
}

inline py::object PyVarSolver::py_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().lyap());
    )
}

inline py::object PyVarSolver::py_t_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().delta_t_abs());
    )
}

inline py::object PyVarSolver::py_delta_s() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().delta_s());
    )
}

py::object PyVarSolver::copy() const{
    return py::cast(PyVarSolver(*this));
}


//===========================================================================================
//                                      PyRK23
//===========================================================================================

PyRK23::PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK23", scalar_type){
}

py::object PyRK23::copy() const{
    return py::cast(PyRK23(*this));
}

//===========================================================================================
//                                      PyRK45
//===========================================================================================

PyRK45::PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK45", scalar_type){}

py::object PyRK45::copy() const{
    return py::cast(PyRK45(*this));
}

//===========================================================================================
//                                      PyDOP853
//===========================================================================================

PyDOP853::PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "DOP853", scalar_type){}

py::object PyDOP853::copy() const{
    return py::cast(PyDOP853(*this));
}

//===========================================================================================
//                                      PyBDF
//===========================================================================================

PyBDF::PyBDF(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(f, jac, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "BDF", scalar_type){}

py::object PyBDF::copy() const{
    return py::cast(PyBDF(*this));
}

//===========================================================================================
//                                      PyRK4
//===========================================================================================

PyRK4::PyRK4(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type) : PySolver(ode, py::none(), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, args, events, "RK4", scalar_type){}

py::object PyRK4::copy() const{
    return py::cast(PyRK4(*this));
}

//===========================================================================================
//                                      PyOdeResult
//===========================================================================================



PyOdeResult::PyOdeResult(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type): DtypeDispatcher(scalar_type), res(result), q0_shape(q0_shape){}


PyOdeResult::PyOdeResult(const PyOdeResult& other) : DtypeDispatcher(other.scalar_type), q0_shape(other.q0_shape) {
        DISPATCH(void, this->res = other.template cast<T>()->clone();)
}


PyOdeResult::PyOdeResult(PyOdeResult&& other) noexcept : DtypeDispatcher(other.scalar_type), res(other.res), q0_shape(std::move(other.q0_shape)) {
    other.res = nullptr;
}


PyOdeResult::~PyOdeResult(){
        DISPATCH(void, delete cast<T>();)
    res = nullptr;
}

PyOdeResult& PyOdeResult::operator=(const PyOdeResult& other){
    if (&other != this){
        DISPATCH(void, delete cast<T>();)
        DISPATCH(void, this->res = other.template cast<T>()->clone();)
        q0_shape = other.q0_shape;
    }
    return *this;
}


PyOdeResult& PyOdeResult::operator=(PyOdeResult&& other) noexcept{
    if (&other != this){
        DISPATCH(void, delete cast<T>();)
        this->res = other.res;
        q0_shape = other.q0_shape;
        other.res = nullptr;
    }
    return *this;
}


py::object PyOdeResult::t() const{
    return DISPATCH(py::object,
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        return py::cast(View<T>(r->t().data(), r->t().size()));
    )
}

py::object PyOdeResult::q() const{
    return DISPATCH(py::object,
        auto *r = reinterpret_cast<OdeResult<T> *>(this->res);
        auto shape = getShape<size_t>(py::ssize_t(r->t().size()), this->q0_shape);
        return py::cast(View<T>(r->q().data(), shape.data(), shape.size()));
    )
}

py::dict PyOdeResult::event_map() const{
    return DISPATCH(py::object,
        EventMap result = reinterpret_cast<const OdeResult<T>*>(this->res)->event_map();
        return to_PyDict(result);
    )
}

py::tuple PyOdeResult::event_data(const py::str& event) const{
    return DISPATCH(py::object,
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        std::vector<T> t_data = r->t_filtered(event.cast<std::string>());
        Array<T> t_res(t_data.data(), t_data.size());
        Array2D<T, 0, 0> q_data = r->q_filtered(event.cast<std::string>());
        auto shape = getShape<size_t>(py::ssize_t(t_res.size()), this->q0_shape);
        Array<T> q_res(q_data.release(), shape.data(), shape.size(), true);
        return py::make_tuple(py::cast(t_res), py::cast(q_res));
    )
}


py::bool_ PyOdeResult::diverges() const{
    return DISPATCH(py::bool_,
        return py::cast(cast<T>()->diverges());
    )
}

py::bool_ PyOdeResult::success() const{
    return DISPATCH(py::bool_,
        return py::cast(cast<T>()->success());
    )
}

py::float_ PyOdeResult::runtime() const{
    return DISPATCH(py::float_,
        return py::cast(cast<T>()->runtime());
    )
}

py::str PyOdeResult::message() const{
    return DISPATCH(py::str,
        return py::cast(cast<T>()->message());
    )
}

void PyOdeResult::examine() const{
    DISPATCH(void,
        return cast<T>()->examine();
    )
}



//===========================================================================================
//                                      PyOdeSolution
//===========================================================================================


PyOdeSolution::PyOdeSolution(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type) : PyOdeResult(result, q0_shape, scalar_type), nsys(prod(q0_shape)) {}

py::object PyOdeSolution::operator()(const py::object& t) const{
    try {
        // Try to convert t to a numpy array
        py::array arr = py::array::ensure(t);
        return DISPATCH(py::object, return this->_get_array<T>(arr);)
    } catch (const py::cast_error&) {
        // If conversion fails, treat as a scalar
        return DISPATCH(py::object, return this->_get_frame<T>(t);)    }
}

//===========================================================================================
//                                      PyODE
//===========================================================================================


PyODE::PyODE(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type) : DtypeDispatcher(scalar_type){
    DISPATCH(void,
        std::vector<T> args;
        OdeData<Func<T>, void> ode_rhs = init_ode_data<T>(data,args, f, py_q0, jacobian, py_args, events);
        std::vector<Event<T>*> safe_events = to_Events<T>(events, shape(py_q0), py_args);
        std::vector<const Event<T>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i];
        }
        auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);

        this->ode = new ODE<T, 0>(ode_rhs, py::cast<T>(t0), q0.data(), q0.size(), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(stepsize), dir, args, evs, method);
        //clean up
        for (size_t i=0; i<evs.size(); i++){
            delete safe_events[i];
        }
    )
}

template<typename T, typename RhsType, typename JacType>
PyODE::PyODE(OdeData<RhsType, JacType> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, const std::vector<const Event<T>*>& events, const std::string& method) : DtypeDispatcher(get_scalar_type<T>()){
    data.is_lowlevel = true;
    data.shape = {py::ssize_t(nsys)};
    this->ode = new ODE<T, 0>(ode, t0, q0, nsys, rtol, atol, min_step, max_step, stepsize, dir, args, events, method);
}

PyODE::PyODE(const PyODE& other) : DtypeDispatcher(other.scalar_type), data(other.data){
    DISPATCH(void, this->ode = other.template cast<T>()->clone();)
    DISPATCH(void, cast<T>()->set_obj(&data);)
}

PyODE::PyODE(PyODE&& other) noexcept : DtypeDispatcher(std::move(other)), ode(other.ode), data(std::move(other.data)){
    other.ode = nullptr;
    DISPATCH(void, cast<T>()->set_obj(&data);)
}

PyODE& PyODE::operator=(const PyODE& other){
    if (&other == this){
        return *this;
    }
    DISPATCH(void, delete cast<T>();)
    DISPATCH(void, this->ode = other.template cast<T>()->clone();)
    data = other.data;
    DISPATCH(void, cast<T>()->set_obj(&data);)
    return *this;
}

PyODE& PyODE::operator=(PyODE&& other) noexcept {
    if (&other == this){
        return *this;
    }
    DISPATCH(void, delete cast<T>();)
    this->ode = other.ode;
    other.ode = nullptr;
    data = std::move(other.data);
    DISPATCH(void, cast<T>()->set_obj(&data);)
    return *this;
}

PyODE::~PyODE(){
    DISPATCH(void, delete cast<T>();)
}

py::object PyODE::call_Rhs(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::object,
        auto q = toCPP_Array<T, Array1D<T>>(py_q);
        if (size_t(q.size()) != cast<T>()->Nsys()){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        const ODE<T>* ode_ptr = this->cast<T>();
        Array1D<T> qdot(ode_ptr->Nsys());
        ode_ptr->Rhs(qdot.data(), py::cast<T>(t), q.data());
        return py::cast(qdot);
    )
}

py::object PyODE::call_Jac(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::object,
        auto q = toCPP_Array<T, Array1D<T>>(py_q);
        if (size_t(q.size()) != cast<T>()->Nsys()){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        const ODE<T>* ode_ptr = this->cast<T>();
        Array2D<T, 0, 0, ndspan::Allocation::Heap, ndspan::Layout::F> jac(ode_ptr->Nsys(), ode_ptr->Nsys());
        ode_ptr->Jac(jac.data(), py::cast<T>(t), q.data(), nullptr);
        for (size_t i=0; i<ode_ptr->Nsys(); i++){
            for (size_t j=i+1; j<ode_ptr->Nsys(); j++){
                T tmp = jac(i, j);
                jac(i, j) = jac(j, i);
                jac(j, i) = tmp;
            }
        }
        return py::cast(View<T, ndspan::Layout::C, 0, 0>(jac.data(), ode_ptr->Nsys(), ode_ptr->Nsys()));
    )
}

py::object PyODE::py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints){

    return DISPATCH(py::object,
        auto* ptr = new OdeResult<T>(cast<T>()->integrate(py::cast<T>(interval), to_step_sequence<T>(t_eval), to_Options(event_options), max_prints));
        return py::cast(PyOdeResult(ptr, this->data.shape, this->scalar_type));
    )
}

py::object PyODE::py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints){
    return DISPATCH(py::object,
        auto* ptr = new OdeSolution<T>(cast<T>()->rich_integrate(py::cast<T>(interval), to_Options(event_options), max_prints));
        return py::cast(PyOdeSolution(ptr, this->data.shape, this->scalar_type));
    )
}

py::object PyODE::py_integrate_until(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints){

    return DISPATCH(py::object,
        auto* ptr = new OdeResult<T>(cast<T>()->integrate_until(py::cast<T>(t), to_step_sequence<T>(t_eval), to_Options(event_options), max_prints));
        return py::cast(PyOdeResult(ptr, this->data.shape, this->scalar_type));
    )
}

py::object PyODE::t_array() const{
    return DISPATCH(py::object,
        auto* r = cast<T>();
        return py::cast(View<T>(r->t().data(), r->t().size()));    
    )
}

py::object PyODE::q_array() const{

    return DISPATCH(py::object,
        auto* r = cast<T>();
        auto shape = getShape<size_t>(py::ssize_t(r->t().size()), this->data.shape);
        return py::cast(View<T>(r->q().data(), shape.data(), shape.size()));
    )
}

py::tuple PyODE::event_data(const py::str& event) const{
    return DISPATCH(py::object,
        std::vector<T> t_data = reinterpret_cast<const ODE<T>*>(ode)->t_filtered(event.cast<std::string>());
        Array2D<T, 0, 0> q_data = reinterpret_cast<const ODE<T>*>(ode)->q_filtered(event.cast<std::string>());
        auto shape = getShape<size_t>(py::ssize_t(t_data.size()), data.shape);
        Array<T> q_res(q_data.release(), shape.data(), shape.size(), true);
        return py::make_tuple(py::cast(Array<T>(t_data.data(), t_data.size())), py::cast(q_res));
    )
}

py::object PyODE::copy() const{
    return py::cast(PyODE(*this));
}

py::object PyODE::solver_copy() const{
    return DISPATCH(py::object,
        auto* ode_ptr = reinterpret_cast<const ODE<T>*>(ode);
        auto* solver_clone = ode_ptr->solver()->clone();
        if (ode_ptr->solver()->method() == "RK45"){
            return py::cast(PyRK45(solver_clone, data, this->scalar_type));
        }else if (ode_ptr->solver()->method() == "DOP853"){
            return py::cast(PyDOP853(solver_clone, data, this->scalar_type));
        }else if (ode_ptr->solver()->method() == "RK23"){
            return py::cast(PyRK23(solver_clone, data, this->scalar_type));
        }else if (ode_ptr->solver()->method() == "BDF"){
            return py::cast(PyBDF(solver_clone, data, this->scalar_type));
        }else if (ode_ptr->solver()->method() == "RK4"){
            return py::cast(PyRK4(solver_clone, data, this->scalar_type));
        }else{
            throw py::value_error("Unregistered solver!");
        }
    )
}

py::dict PyODE::event_map() const{
    return DISPATCH(py::dict,
        EventMap result = cast<T>()->event_map();
        return to_PyDict(result);
    )
}

py::object PyODE::Nsys() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->Nsys());
    )
}

py::object PyODE::runtime() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->runtime());
    )
}

py::object PyODE::diverges() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->diverges());
    )
}

py::object PyODE::is_dead() const{
    return DISPATCH(py::object,
        return py::cast(cast<T>()->is_dead());
    )
}

void PyODE::reset() {
    DISPATCH(void,
        return cast<T>()->reset();
    )
}

void PyODE::clear() {
    DISPATCH(void,
        return cast<T>()->clear();
    )
}


//===========================================================================================
//                                      PyVarODE
//===========================================================================================


PyVarODE::PyVarODE(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& period, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type):PyODE(scalar_type){
    DISPATCH(void,
        std::vector<T> args;
        OdeData<Func<T>, void> ode_rhs = init_ode_data<T>(this->data, args, f, q0, jac, py_args, events);
        Array1D<T> q0_ = toCPP_Array<T, Array1D<T>>(q0);
        if ((q0_.size() & 1) != 0){
            throw py::value_error("Variational ODEs require an even number of system size");
        }
        std::vector<Event<T>*> safe_events = to_Events<T>(events, shape(q0), py_args);
        std::vector<const Event<T>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i];
        }

        this->ode = new VariationalODE<T, 0, Func<T>, void>(ode_rhs, py::cast<T>(t0), q0_.data(), q0_.size()/2, py::cast<T>(period), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(stepsize), dir, args, evs, method.cast<std::string>());
        for (size_t i=0; i<evs.size(); i++){
            delete safe_events[i];
        }
    )
}

py::object PyVarODE::py_t_lyap() const{
    return DISPATCH(py::object,
        const auto& vode = varode<T>();
        View<T> res(vode.t_lyap().data(), vode.t_lyap().size());
        return py::cast(res);
    )
}

py::object PyVarODE::py_lyap() const{

    return DISPATCH(py::object,
        const auto& vode = varode<T>();
        View<T> res(vode.lyap().data(), vode.t_lyap().size());
        return py::cast(res);
    )
}

py::object PyVarODE::py_kicks() const{
    return DISPATCH(py::object,
        const auto& vode = varode<T>();
        View<T> res(vode.kicks().data(), vode.t_lyap().size());
        return py::cast(res);
    )
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


void py_integrate_all(py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress){
    // Separate lists for each numeric type
    std::vector<void*> array;
    std::vector<int> types;

    std::vector<void*> step_seq;


    // Iterate through the list and identify each PyODE type
    for (const py::handle& item : list) {
        try {
            auto& pyode = item.cast<PyODE&>();

            // Use the scalar_type to determine which array to add to
            if (!pyode.data.is_lowlevel) {
                throw py::value_error("All ODE's in integrate_all must use only compiled functions, and no pure python functions");
            }
            array.push_back(pyode.ode);
            types.push_back(pyode.scalar_type);
            if (size_t(pyode.scalar_type) >= step_seq.size()){
                step_seq.resize(pyode.scalar_type+1);
                
                step_seq[pyode.scalar_type] = call_dispatch(pyode.scalar_type, [&]<typename T>() -> void* {
                    return new StepSequence<T>(to_step_sequence<T>(t_eval));
                });


            }
        } catch (const py::cast_error&) {
            // If cast failed, throw an error
            throw py::value_error("List item is not a recognized PyODE object type.");
        }
    }

    auto options = to_Options(event_options);

    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    int tot = 0;
    const int target = int(array.size());
    Clock clock;
    clock.start();

    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (size_t i=0; i<array.size(); i++){

        call_dispatch(types[i], [&]<typename T>() LAMBDA_INLINE {
            ODE<T>* ode = reinterpret_cast<ODE<T>*>(array[i]);
            ode->integrate(T(interval), *reinterpret_cast<StepSequence<T>*>(step_seq[types[i]]), options);
        });

        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }

    for (size_t i=0; i<step_seq.size(); i++){
        call_dispatch(int(i), [&]<typename T>(){
            delete reinterpret_cast<StepSequence<T>*>(step_seq[i]);
        });
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;
}

void py_advance_all(py::object& list, double t_goal, int threads, bool display_progress){
    // Separate lists for each numeric type
    std::vector<void*> array;
    std::vector<int> types;

    // Iterate through the list and identify each PySolver type
    for (const py::handle& item : list) {
        try {
            auto& pysolver = item.cast<PySolver&>();


            if (!pysolver.data.is_lowlevel) {
                throw py::value_error("All solvers in advance_all must use only compiled functions, and no pure python functions");
            }
            array.push_back(pysolver.s);
            types.push_back(pysolver.scalar_type);
        } catch (const py::cast_error&) {
            // If cast failed, throw an error
            throw py::value_error("List item is not a recognized PySolver object type.");
        }
    }

    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    int tot = 0;
    const int target = int(array.size());
    Clock clock;
    clock.start();

    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (size_t i=0; i<array.size(); i++){

        call_dispatch(types[i], [&]<typename T>() LAMBDA_INLINE {
            auto* solver = reinterpret_cast<OdeSolver<T>*>(array[i]);
            solver->advance_until(T(t_goal));
        });

        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;

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
        .def(py::init<std::string, py::object, py::object, bool, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("period", &PyPerEvent::period);


    py::class_<PySolver>(m, "OdeSolver")
        .def_property_readonly("t", &PySolver::t)
        .def_property_readonly("q", &PySolver::q)
        .def_property_readonly("t_old", &PySolver::t_old)
        .def_property_readonly("q_old", &PySolver::q_old)
        .def_property_readonly("stepsize", &PySolver::stepsize)
        .def_property_readonly("diverges", &PySolver::diverges)
        .def_property_readonly("is_dead", &PySolver::is_dead)
        .def_property_readonly("Nsys", &PySolver::Nsys)
        .def_property_readonly("n_evals_rhs", &PySolver::n_evals_rhs)
        .def_property_readonly("status", &PySolver::message)
        .def_property_readonly("at_event", &PySolver::py_at_event)
        .def("event_located", &PySolver::py_event_located, py::arg("event"))
        .def("show_state", &PySolver::show_state,
            py::arg("digits") = 8
        )
        .def("advance", &PySolver::advance)
        .def("advance_to_event", &PySolver::advance_to_event)
        .def("advance_until", &PySolver::advance_until, py::arg("t"))
        .def("reset", &PySolver::reset)
        .def("set_ics", &PySolver::set_ics, py::arg("t0"), py::arg("q0"), py::arg("stepsize")=0, py::arg("direction")=0)
        .def("resume", &PySolver::resume)
        .def("copy", &PySolver::copy)
        .def_property_readonly("scalar_type", [](const PySolver& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyRK23, PySolver>(m, "RK23")
        .def(py::init<PyRK23>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyRK45, PySolver>(m, "RK45")
        .def(py::init<PyRK45>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyDOP853, PySolver>(m, "DOP853")
        .def(py::init<PyDOP853>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyBDF, PySolver>(m, "BDF")
        .def(py::init<PyBDF>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");


    py::class_<PyRK4, PySolver>(m, "RK4")
        .def(py::init<PyRK4>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyVarSolver, PySolver>(m, "VariationalSolver")
        .def(py::init<PyVarSolver>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, std::string, std::string>(),
            py::arg("f"),
            py::arg("jac"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def_property_readonly("logksi", &PyVarSolver::py_logksi)
        .def_property_readonly("lyap", &PyVarSolver::py_lyap)
        .def_property_readonly("t_lyap", &PyVarSolver::py_t_lyap)
        .def_property_readonly("delta_s", &PyVarSolver::py_delta_s);

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
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyODE>(), py::arg("ode"))
        .def("rhs", &PyODE::call_Rhs, py::arg("t"), py::arg("q"))
        .def("jac", &PyODE::call_Jac, py::arg("t"), py::arg("q"))
        .def("solver", &PyODE::solver_copy, py::keep_alive<0, 1>())
        .def("integrate", &PyODE::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("integrate_until", &PyODE::py_integrate_until,
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
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyVarODE>(), py::arg("ode"))
        .def_property_readonly("t_lyap", &PyVarODE::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE::py_lyap)
        .def_property_readonly("kicks", &PyVarODE::py_kicks)
        .def("copy", &PyVarODE::copy);

    py::class_<PyVecFieldBase>(m, "SampledVectorField")
        .def("streamline", &PyVecFieldBase::py_streamline,
            py::arg("q0"),
            py::arg("length"),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("t_eval")=py::none(),
            py::arg("method")="RK45"
        )
        .def("get_ode", &PyVecFieldBase::py_streamline_ode,
            py::arg("q0"),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("method")="RK45",
            py::arg("normalized")=false, py::keep_alive<0, 1>()
        )
        .def("streamplot_data", &PyVecFieldBase::py_streamplot_data,
            py::arg("max_length"),
            py::arg("ds"),
            py::arg("density")=30
        );

    py::class_<PyVecField2D, PyVecFieldBase>(m, "SampledVectorField2D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("vx"),
            py::arg("vy"))
        .def_property_readonly("x", &PyVecField2D::py_x<0>)
        .def_property_readonly("y", &PyVecField2D::py_x<1>)
        .def_property_readonly("vx", &PyVecField2D::py_vx<0>)
        .def_property_readonly("vy", &PyVecField2D::py_vx<1>)
        .def("get_vx", &PyVecField2D::py_vx_at<0, double, double>, py::arg("x"), py::arg("y"))
        .def("get_vy", &PyVecField2D::py_vx_at<1, double, double>, py::arg("x"), py::arg("y"))
        .def("__call__", &PyVecField2D::py_vector<double, double>, py::arg("x"), py::arg("y"))
        .def("in_bounds", &PyVecField2D::py_in_bounds<double, double>, py::arg("x"), py::arg("y"));

    py::class_<PyVecField3D, PyVecFieldBase>(m, "SampledVectorField3D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("vx"),
            py::arg("vy"),
            py::arg("vz"))
        .def_property_readonly("x", &PyVecField3D::py_x<0>)
        .def_property_readonly("y", &PyVecField3D::py_x<1>)
        .def_property_readonly("z", &PyVecField3D::py_x<2>)
        .def_property_readonly("vx", &PyVecField3D::py_vx<0>)
        .def_property_readonly("vy", &PyVecField3D::py_vx<1>)
        .def_property_readonly("vz", &PyVecField3D::py_vx<2>)
        .def("get_vx", &PyVecField3D::py_vx_at<0, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("get_vy", &PyVecField3D::py_vx_at<1, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("__call__", &PyVecField3D::py_vector<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("in_bounds", &PyVecField3D::py_in_bounds<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"));

    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

    m.def("advance_all", &py_advance_all, py::arg("solvers"), py::arg("t_goal"), py::arg("threads")=-1, py::arg("display_progress")=false);

#ifdef MPREAL
    m.def("set_mpreal_prec",
      &mpfr::mpreal::set_default_prec,
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", &mpfr::mpreal::get_default_prec);
#else
    m.def("set_mpreal_prec",
      [](size_t){
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      });
#endif
}

}
