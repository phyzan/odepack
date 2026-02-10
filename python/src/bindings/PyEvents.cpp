#include "../../../include/odepack.hpp"
#include "odetemplates.hpp"

namespace ode{

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
//                                      Helper functions
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

} // namespace ode