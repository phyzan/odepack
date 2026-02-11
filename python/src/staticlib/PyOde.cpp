#include "../../../include/pyode/lib_impl/PyOde_impl.hpp"
#include "../../../include/pyode/lib/PySolver.hpp"
#include "../../../include/pyode/lib/PySubSolver.hpp"
#include "../../../include/pyode/lib/PyEvents.hpp"
#include "../../../include/pyode/lib/PyResult.hpp"


namespace ode{

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

PyODE::PyODE(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}

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
//                                      Additional functions
//===========================================================================================

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


#define DEFINE_ODE(T) \
    template class ODE<T, 0>; \
    template class EventCounter<T, 0>; \
    template void integrate_all(const std::vector<ODE<T, 0>*>&, const T&, const StepSequence<T>&, const std::vector<EventOptions>&, int, bool); \
    template void ODE<T, 0>::_init<Func<T>, void>(OdeData<Func<T>, void> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, const std::vector<const Event<T>*>& events, const std::string& method); \
    template PyODE::PyODE(OdeData<Func<T>, void> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args={}, const std::vector<const Event<T>*>& events, const std::string&); \

DEFINE_ODE(float)
DEFINE_ODE(double)
DEFINE_ODE(long double)
#ifdef MPREAL
DEFINE_ODE(mpfr::mpreal)
#endif

#undef DEFINE_ODE

} // namespace ode