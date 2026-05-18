#ifndef PYSOLVER_IMPL_HPP
#define PYSOLVER_IMPL_HPP

#include "../lib/PySolver.hpp"
#include "../lib/PyTools.hpp"
#include "../lib/PyEvents.hpp"
#include "../pycast/pycast.hpp"
#include <atomic>

namespace ode{

template<typename T>
void PyConstSolver::init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
    std::vector<T> args;
    init_ode_data<T, false>([&](const auto& ode_obj){
        std::vector<std::unique_ptr<Event<T>>> safe_events = to_Events<T>(py_events, this->data.shape, py_args);
        std::vector<const Event<T>*> evs(safe_events.size());
        for (size_t i=0; i<evs.size(); i++){
            evs[i] = safe_events[i].get();
        }
        auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
        this->s = get_virtual_solver<T, 0>(getIntegrator(name), ode_obj, py::cast<T>(t0), q0.data(), q0.size(), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(stepsize), dir, args, evs).release();

    }, this->data, args, f, py_q0, jac, py_args, py_events);
}

template<typename T>
OdeRichSolver<T>* PyConstSolver::cast(){
    return reinterpret_cast<OdeRichSolver<T>*>(this->s);
}

template<typename T>
const OdeRichSolver<T>* PyConstSolver::cast()const{
    return reinterpret_cast<const OdeRichSolver<T>*>(this->s);
}


template<typename T, bool FORCE_JAC, typename Callable>
void init_ode_data(Callable&& action, PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    // passes a copy of the final form of data to the OdeData object. As a result, the referenced data object should not be modified for the entire lifetime of the solver constructed with its copy.
    std::string scalar_type = get_scalar_type<T>();
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);

    bool f_is_compiled = py::isinstance<PyFuncWrapper>(f) || py::isinstance<py::capsule>(f);
    bool jac_is_compiled = !jacobian.is_none() && (py::isinstance<PyFuncWrapper>(jacobian) || py::isinstance<py::capsule>(jacobian));
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    PyRhsFunc<T> rhs = nullptr;
    PyRhsFunc<T> jac = nullptr;
    if (f_is_compiled){
        if (py::isinstance<PyFuncWrapper>(f)){
            //safe approach
            auto& _f = f.cast<PyFuncWrapper&>();
            rhs = reinterpret_cast<PyRhsFunc<T>>(_f.rhs);
            if (_f.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size");
            }
            else if (_f.Nargs != args.size()){
                throw py::value_error("The number of the provided extra args (" + std::to_string(args.size()) + ") differs from the number of args specified for this ode system ("+std::to_string(_f.Nargs)+").");
            }
        }
        else{
            rhs = open_capsule<PyRhsFunc<T>>(f.cast<py::capsule>());
        }
    }
    else{
        data.rhs = f;
        rhs = py_rhs;
    }
    if (jac_is_compiled){
        if (py::isinstance<PyFuncWrapper>(jacobian)){
            //safe approach
            auto& _j = jacobian.cast<PyFuncWrapper&>();
            jac = reinterpret_cast<PyRhsFunc<T>>(_j.rhs);
            if (_j.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size that applied in the provided jacobian");
            } else if (_j.Nargs != args.size()){
                throw py::value_error("The array size of the given extra args differs from the number of args specified for the provided jacobian");
            }
        }
        else{
            jac = reinterpret_cast<PyRhsFunc<T>>(open_capsule<PyRhsFunc<T>>(jacobian.cast<py::capsule>()));
        }
    }else if (!jacobian.is_none()){
        data.jac = jacobian;
        jac = reinterpret_cast<PyRhsFunc<T>>(py_jac<T>);
    }
    
    if constexpr (FORCE_JAC){
        if (jac == nullptr){
            throw py::value_error("No jacobian was provided, but the solver requires one");
        }
    }

    for (py::handle ev : events){
        if (!py::isinstance<PyEvent>(ev)) {
            throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class. Instance of type '" + std::string(py::str(py::type::of(ev))) + "' was found.");
        }
        const auto& _ev = ev.cast<const PyEvent&>();
        if (_ev.scalar_type != DTYPE_MAP.at(scalar_type)){
            throw py::value_error("All event objects in 'events' must have scalar type " + scalar_type + ".");
        }
        _ev.check_sizes(_size, args.size());
    }

    // allocate dummy array in PyStruct data
    if constexpr (std::is_floating_point_v<T>) {
        py::array_t<T>& array = data.template get_array<T>();
        array.resize({static_cast<py::ssize_t>(_size)});
    }
    data.is_lowlevel = f_is_compiled && (jac_is_compiled || jacobian.is_none()) && all_are_lowlevel(events);

    if constexpr (FORCE_JAC){
        // pass a jacobian whose type is not nullptr at compile time
        action(OdeData{.Rhs = [rhs, pydata=data](T* res, const T& t, const T* q, const T* args){
            rhs(res, t, q, args, &pydata);
        }, .Jac = [jac, pydata=data](T* res, const T& t, const T* q, const T* args){
            jac(res, t, q, args, &pydata);
        }});
    } else if (jac == nullptr){
        // pass a jacobian whose type is std::nullptr_t at compile time, forcing the solver to use finite differences for jacobian approximation
        action(OdeData{.Rhs = [rhs, pydata=data](T* res, const T& t, const T* q, const T* args){
            rhs(res, t, q, args, &pydata);
        }, .Jac = nullptr});
    } else {
        action(OdeData{.Rhs = [rhs, pydata=data](T* res, const T& t, const T* q, const T* args){
            rhs(res, t, q, args, &pydata);
        }, .Jac = [jac, pydata=data](T* res, const T& t, const T* q, const T* args){
            jac(res, t, q, args, &pydata);
        }});
    }



}

template<typename Callable>
void py_advance_all_general(py::object& list, Callable&& func, int threads, bool display_progress){
    // Separate lists for each numeric type
    std::vector<void*> array;
    std::vector<ScalarType> types;

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

    std::exception_ptr thread_exception = nullptr;
    std::atomic<bool> error_flag{false};

    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (size_t i=0; i<array.size(); i++){
        if (error_flag.load()){ continue;}
        try {
            call_dispatch(types[i], [&]<typename T>() LAMBDA_INLINE {
                auto* solver = reinterpret_cast<OdeRichSolver<T>*>(array[i]);
                func(solver);
            });
        } catch (...) {
            #pragma omp critical
            {
                if (!error_flag.load()) {
                    thread_exception = std::current_exception();
                    error_flag.store(true);
                }
            }
            continue;
        }

        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;

}

}

#endif // PYSOLVER_IMPL_HPP