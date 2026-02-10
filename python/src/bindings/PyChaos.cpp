#include "../../../include/odepack.hpp"
#include "odetemplates.hpp"


namespace ode{

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

py::object PyVarSolver::py_logksi() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().logksi());
    )
}

py::object PyVarSolver::py_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().lyap());
    )
}

py::object PyVarSolver::py_t_lyap() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().delta_t_abs());
    )
}

py::object PyVarSolver::py_delta_s() const{
    return DISPATCH(py::object,
        return py::cast(this->main_event<T>().delta_s());
    )
}

py::object PyVarSolver::copy() const{
    return py::cast(PyVarSolver(*this));
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


} // namespace ode