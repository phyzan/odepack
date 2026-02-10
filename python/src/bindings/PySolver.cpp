#include "../../../include/odepack.hpp"
#include "odetemplates.hpp"

namespace ode{

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
//                                      Additional functions
//===========================================================================================


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

} // namespace ode