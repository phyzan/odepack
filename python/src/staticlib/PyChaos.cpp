#include "../../../include/pyodepack.hpp"



namespace ode{

//===========================================================================================
//                                      PyVarSolver
//===========================================================================================


PyVarSolver::PyVarSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::iterable& py_delta_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const std::string& method, const std::string& scalar_type) : PySolver(scalar_type) {
    DISPATCH(void,
        if (jac.is_none()){
            throw py::value_error("Variational solvers require an exact jacobian for the original system");
        }
        size_t nsys = size_t(py::len(py_q0));
        if (size_t(py::len(py_delta_q0)) != nsys){
            throw py::value_error("The variational state vector delta_q0 must have the same size as q0");
        }

        // ------- fill the initial state vector with q0 and delta_q0 -------
        Array1D<T> vector(2*nsys);
        int i = 0;
        for (auto item : py_q0) {
            // Cast Python object to double safely
            auto val = py::cast<T>(item);
            // Construct T from double
            vector[i] = val;
            i++;
        }
        for (auto item : py_delta_q0) {
            // Cast Python object to double safely
            auto val = py::cast<T>(item);
            // Construct T from double
            vector[i] = val;
            i++;
        }
        // ----------------------------------------------------------

        std::vector<T> args;
        init_ode_data<T, true>([&](const auto& ode_obj){


            std::vector<std::unique_ptr<Event<T>>> safe_events = to_Events<T>(events, shape(py_q0), py_args);
            std::vector<const Event<T>*> evs(safe_events.size());
            for (size_t j=0; j<evs.size(); j++){
                evs[j] = safe_events[j].get();
            }

            this->s = get_virtual_variational_solver<T, 0>(getIntegrator(method), period.cast<T>(), ode_obj, t0.cast<T>(), vector.data(), vector.data()+nsys, nsys, rtol.cast<T>(), atol.cast<T>(), min_step.cast<T>(), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), stepsize.cast<T>(), dir, args, evs).release();
        }, this->data, args, f, py_q0, jac, py_args, events);


    )
}

PyVarSolver::PyVarSolver(void* solver, PyStruct py_data, ScalarType scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

py::object PyVarSolver::py_logksi() const{
    return DISPATCH(py::object,
        return py::cast(this->cast<T>()->log_ksi());
    )
}

py::object PyVarSolver::py_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->cast<T>()->lyapunov_exponent());
    )
}

py::object PyVarSolver::py_t_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->cast<T>()->elapsed_time());
    )
}

py::object PyVarSolver::py_delta_s() const{
    return DISPATCH(py::object,
        return py::cast(this->cast<T>()->stretching_number());
    )
}

py::object PyVarSolver::copy() const{
    return py::cast(PyVarSolver(*this));
}




//===========================================================================================
//                                      PyVarODE
//===========================================================================================


PyVarODE::PyVarODE(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& q0_main, const py::iterable& delta_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type) : PyODE(scalar_type){
    DISPATCH(void,
        if (jac.is_none()){
            throw py::value_error("Variational solvers require an exact jacobian for the original system");
        }
        size_t nsys = size_t(py::len(q0_main));
        if (size_t(py::len(delta_q0)) != nsys){
            throw py::value_error("The variational state vector delta_q0 must have the same size as q0");
        }

        // ------- fill the initial state vector with q0 and delta_q0 -------
        Array1D<T> vector(2*nsys);
        int i = 0;
        for (auto item : q0_main) {
            // Cast Python object to double safely
            auto val = py::cast<T>(item);
            // Construct T from double
            vector[i] = val;
            i++;
        }
        for (auto item : delta_q0) {
            // Cast Python object to double safely
            auto val = py::cast<T>(item);
            // Construct T from double
            vector[i] = val;
            i++;
        }
        // ----------------------------------------------------------
        
        //Join q0_main and delta_q0 into a python iterable q0 that we can pass to init_ode_data, from the existing vector object that we defined.
        py::list q0;
        for (size_t j=0; j<2*nsys; j++){
            q0.append(py::cast(vector[j]));
        }

        std::vector<T> args;
        init_ode_data<T, true>([&](const auto& ode_obj){
            std::vector<std::unique_ptr<Event<T>>> safe_events = to_Events<T>(events, shape(q0), py_args);
            std::vector<const Event<T>*> evs(safe_events.size());
            for (size_t j=0; j<evs.size(); j++){
                evs[j] = safe_events[j].get();
            }

            this->ode = new VariationalODE<T, 0>(ode_obj, py::cast<T>(t0), vector.data(), vector.data()+nsys, nsys, py::cast<T>(period), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(stepsize), dir, args, evs, getIntegrator(method));

            }, this->data, args, f, q0, jac, py_args, events);
    )
}

py::object PyVarODE::py_t_lyap() const{
    return DISPATCH(py::object,
        const VariationalODE<T, 0>& vode = varode<T>();
        View<T> res(vode.renorm_times().data(), vode.renorm_times().size());
        return py::cast(res);
    )
}

py::object PyVarODE::py_lyap() const{

    return DISPATCH(py::object,
        const VariationalODE<T, 0>& vode = varode<T>();
        View<T> res(vode.lyap_values().data(), vode.lyap_values().size());
        return py::cast(res);
    )
}

py::object PyVarODE::py_kicks() const{
    return DISPATCH(py::object,
        const VariationalODE<T, 0>& vode = varode<T>();
        View<T> res(vode.kick_values().data(), vode.kick_values().size());
        return py::cast(res);
    )
}

py::object PyVarODE::copy() const{
    return py::cast(PyVarODE(*this));
}


//===========================================================================================
//                                EXPLICIT TEMPLATE INSTANTIATIONS
//===========================================================================================




} // namespace ode