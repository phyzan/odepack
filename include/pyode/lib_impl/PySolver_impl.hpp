#ifndef PYSOLVER_IMPL_HPP
#define PYSOLVER_IMPL_HPP

#include "../lib/PySolver.hpp"
#include "../lib/PyTools.hpp"
#include "../lib/PyEvents.hpp"
#include "../pycast/pycast.hpp"

namespace ode{

template<typename T>
void PySolver::init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name){
    std::vector<T> args;
    OdeData<Func<T>, void> ode_data = init_ode_data<T>(this->data, args, f, py_q0, jac, py_args, py_events);
    std::vector<Event<T>*> safe_events = to_Events<T>(py_events, this->data.shape, py_args);
    std::vector<const Event<T>*> evs(safe_events.size());
    for (size_t i=0; i<evs.size(); i++){
        evs[i] = safe_events[i];
    }
    auto q0 = toCPP_Array<T, Array1D<T>>(py_q0);
    this->s = get_virtual_solver<T, 0>(name, ode_data, py::cast<T>(t0), q0.data(), q0.size(), py::cast<T>(rtol), py::cast<T>(atol), py::cast<T>(min_step), (max_step.is_none() ? inf<T>() : max_step.cast<T>()), py::cast<T>(stepsize), dir, args, evs).release();
    for (size_t i=0; i<evs.size(); i++){
        delete safe_events[i];
    }
}

template<typename T>
OdeRichSolver<T>* PySolver::cast(){
    return reinterpret_cast<OdeRichSolver<T>*>(this->s);
}

template<typename T>
const OdeRichSolver<T>* PySolver::cast()const{
    return reinterpret_cast<const OdeRichSolver<T>*>(this->s);
}


template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    std::string scalar_type = get_scalar_type<T>();
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);

    bool f_is_compiled = py::isinstance<PyFuncWrapper>(f) || py::isinstance<py::capsule>(f);
    bool jac_is_compiled = !jacobian.is_none() && (py::isinstance<PyFuncWrapper>(jacobian) || py::isinstance<py::capsule>(jacobian));
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<Func<T>, void> ode_rhs = {nullptr, nullptr, nullptr};
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
        data.rhs = f;
        ode_rhs.rhs = py_rhs;
    }
    if (jac_is_compiled){
        if (py::isinstance<PyFuncWrapper>(jacobian)){
            //safe approach
            auto& _j = jacobian.cast<PyFuncWrapper&>();
            ode_rhs.jacobian = (VoidType)_j.rhs;
            if (_j.Nsys != _size){
                throw py::value_error("The array size of the initial conditions differs from the ode system size that applied in the provided jacobian");
            }
            else if (_j.Nargs != args.size()){
                throw py::value_error("The array size of the given extra args differs from the number of args specified for the provided jacobian");
            }
        }
        else{
            ode_rhs.jacobian = (VoidType)open_capsule<Func<T>>(jacobian.cast<py::capsule>());
        }
    }else if (!jacobian.is_none()){
        data.jac = jacobian;
        ode_rhs.jacobian = (VoidType)py_jac<T>;
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
    if (!data.is_lowlevel){
        ode_rhs.obj = &data;
    }
    return ode_rhs;
}

}

#endif // PYSOLVER_IMPL_HPP