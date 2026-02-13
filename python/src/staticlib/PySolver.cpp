#include "../../../include/pyode/lib_impl/PySolver_impl.hpp"
#include "../../../include/ode/SolverDispatcher_impl.hpp"
#include "../../../include/ode/Solvers/Solvers_impl.hpp"
#include "../../../include/ode/Core/SolverBase_impl.hpp"
#include "../../../include/ode/SolverState_impl.hpp"
#include "../../../include/ode/Core/RichBase_impl.hpp"

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
    set_pyobj(*this);
}

PySolver::PySolver(const PySolver& other) : DtypeDispatcher(other), data(other.data){
    DISPATCH(void, this->s = other.template cast<T>()->clone();)
    set_pyobj(other);
}

PySolver::PySolver(PySolver&& other) noexcept : DtypeDispatcher(std::move(other)), s(other.s), data(std::move(other.data)){
    other.s = nullptr;
    set_pyobj(other);
}


PySolver& PySolver::operator=(const PySolver& other){
    if (&other != this){
        data = other.data;
        DISPATCH(void, delete cast<T>();)
        DISPATCH(void, this->s = other.template cast<T>()->clone();)
        set_pyobj(other);
    }
    return *this;
}


PySolver& PySolver::operator=(PySolver&& other) noexcept{
    if (&other != this){
        data = std::move(other.data);
        DISPATCH(void, delete cast<T>();)
        this->s = other.s;
        set_pyobj(other);
        other.s = nullptr;
    }
    return *this;
}


PySolver::~PySolver(){
    DISPATCH(void, delete cast<T>();)
}


void PySolver::set_pyobj(const PySolver& other){
    if (!data.is_lowlevel){
        DISPATCH(void, cast<T>()->set_obj(&data);)
    }
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

py::object PySolver::n_evals_jac() const{
    return DISPATCH(py::object, return py::cast(cast<T>()->n_evals_jac()); )
}


void PySolver::show_state(int digits) const{
    DISPATCH(void,
        return cast<T>()->show_state(digits);
    )
}

py::object PySolver::py_rhs(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::object,
        size_t nsys = cast<T>()->Nsys();
        Array1D<T> tmp(2*nsys);
        if (size_t(py::len(py_q)) != nsys){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        pass_values(tmp.data()+nsys, py_q, nsys);
        cast<T>()->Rhs(tmp.data(), py::cast<T>(t), tmp.data()+nsys);
        return py::cast(View1D<T>(tmp.data(), nsys));
    )
}

py::object PySolver::py_jac(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::object,
        size_t nsys = cast<T>()->Nsys();
        Array1D<T> q(nsys);
        Array2D<T> jac(nsys, nsys);
        if (size_t(py::len(py_q)) != nsys){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        pass_values(q.data(), py_q, nsys);
        cast<T>()->Jac(jac.data(), t.cast<T>(), q.data(), nullptr);
        for (size_t i=0; i<nsys; i++){
            for (size_t j=i+1; j<nsys; j++){
                std::swap(jac(i, j), jac(j, i));
            }
        }
        return py::cast(jac);
    )
}

py::tuple PySolver::timeit_rhs(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::tuple,
        size_t nsys = cast<T>()->Nsys();
        Array1D<T> tmp(2*nsys);
        if (size_t(py::len(py_q)) != nsys){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        pass_values(tmp.data()+nsys, py_q, nsys);
        auto start = std::chrono::high_resolution_clock::now();
        cast<T>()->Rhs(tmp.data(), py::cast<T>(t), tmp.data()+nsys);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        return py::make_tuple(py::cast(duration.count()), py::cast(View1D<T>(tmp.data(), nsys)));
    )
}


py::tuple PySolver::timeit_jac(const py::object& t, const py::iterable& py_q) const{
    return DISPATCH(py::tuple,
        size_t nsys = cast<T>()->Nsys();
        Array1D<T> q(nsys);
        Array2D<T> jac(nsys, nsys);
        if (size_t(py::len(py_q)) != nsys){
            throw py::value_error("Invalid size of state array in call to rhs");
        }
        pass_values(q.data(), py_q, nsys);
        auto start = NOW;
        cast<T>()->Jac(jac.data(), t.cast<T>(), q.data(), nullptr);
        auto end = NOW;
        std::chrono::duration<double, std::milli> duration = end - start;
        for (size_t i=0; i<nsys; i++){
            for (size_t j=i+1; j<nsys; j++){
                std::swap(jac(i, j), jac(j, i));
            }
        }
        return py::make_tuple(py::cast(duration.count()), py::cast(jac));
    )
}

py::object PySolver::advance() {
    return DISPATCH(py::object,
        return py::cast(cast<T>()->advance());
    )
}

py::tuple PySolver::timeit_step() {
    return DISPATCH(py::tuple,
        auto start = NOW;
        bool success = cast<T>()->advance();
        auto end = NOW;
        std::chrono::duration<double, std::milli> duration = end - start;
        return py::make_tuple(py::cast(duration.count()), py::cast(success));
    )
}

py::object PySolver::advance_to_event() {
    return DISPATCH(py::object,
        return py::cast(cast<T>()->advance_to_event());
    )
}

py::object PySolver::advance_until(const py::object& time, const py::object& observer) {
    return DISPATCH(py::object,
        if (observer.is_none()){
            return py::cast(cast<T>()->advance_until(time.cast<T>()));
        }
        py::function py_obs;
        try{
            py_obs = observer.cast<py::function>();
        } catch (const py::cast_error&){
            throw py::value_error("The observer parameter must be a function that takes no arguments");
        }
        std::function<void(const T&, const T*)> obs = [py_obs](const T&, const T*){
            py_obs();
        };
        OdeRichSolver<T>* solver = this->cast<T>();
        return py::cast(solver->observe_until(time.cast<T>(), obs));
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

void PySolver::stop(const py::str& reason) { DISPATCH(void, cast<T>()->stop(reason.cast<std::string>()); ) } void PySolver::kill(const py::str& reason) { DISPATCH(void, cast<T>()->kill(reason.cast<std::string>()); ) }

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

// Explicit instantiations for virtual solver factory.
namespace ode{

#define ODEPACK_INSTANTIATE_VIRTUAL_SOLVER(T) \
    template std::unique_ptr<OdeRichSolver<T, 0>> \
    get_virtual_solver<T, 0, Func<T>, void>(const std::string& name, \
        OdeData<Func<T>, void> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, \
        T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, \
        const std::vector<const Event<T>*>& events); \
    template struct SolverState<T, 0>; \
    template struct SolverRichState<T, 0>; \
    template RK4<T, 0, SolverPolicy::RichVirtual, Func<T>, void>::RK4(OdeData<Func<T>, void> ode, T t0, const T* q0, size_t nsys, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, const std::vector<T>& args, const std::vector<const Event<T>*>& events); \

ODEPACK_INSTANTIATE_VIRTUAL_SOLVER(float)
ODEPACK_INSTANTIATE_VIRTUAL_SOLVER(double)
ODEPACK_INSTANTIATE_VIRTUAL_SOLVER(long double)
#ifdef MPREAL
ODEPACK_INSTANTIATE_VIRTUAL_SOLVER(mpfr::mpreal)
#endif

#undef ODEPACK_INSTANTIATE_VIRTUAL_SOLVER

} // namespace ode