#ifndef PY_EVENTS_IMPL_HPP
#define PY_EVENTS_IMPL_HPP

#include "../lib/PyEvents.hpp"
#include "../../ode/Core/Events_impl.hpp"
#include "../pycast/pycast.hpp"

namespace ode{


template<typename T>
auto PyEvent::mask() const{
    return [pydata=this->data, mask=this->_mask, use_pymask=this->_py_mask](T* out, const T& t, const T* q, const T* args){
        if (use_pymask){
            return py_mask<T>(out, t, q, args, &pydata);
        } else {
            assert(mask != nullptr && "mask is null");
            return reinterpret_cast<RhsFunc<T>>(mask)(out, t, q, args);
        }
    };
}

template<typename T>
ObjFun<T> PyPrecEvent::when() const{
    if (_when == nullptr){
        return py_event<T>;
    }
    else{
        return reinterpret_cast<ObjFun<T>>(this->_when);
    }
}

template<typename T>
void* PyPrecEvent::get_new_event(){
    auto target = [pydata=this->data, target=_when](const T& t, const T* q, const T* args){
        if (target == nullptr){
            return pydata.event(t, py::cast(View<T, ndspan::Layout::C>(q, pydata.shape.data(), pydata.shape.size())), *pydata.py_args).template cast<T>();
        } else {
            return reinterpret_cast<ObjFun<T>>(target)(t, q, args);
        }
    };

    if (this->_py_mask || this->_mask != nullptr){
        return new PreciseEvent(this->name().cast<std::string>(), target, this->_event_tol.cast<T>(), _dir, this->mask<T>(), this->hide_mask());
    } else {
        return new PreciseEvent(this->name().cast<std::string>(), target, this->_event_tol.cast<T>(), _dir, nullptr, this->hide_mask());
    }
}

template<typename T>
void* PyPerEvent::get_new_event(){
    if (this->_py_mask || this->_mask != nullptr){
        return new PeriodicEvent(this->name().cast<std::string>(), _period.cast<T>(), this->mask<T>(), this->hide_mask());
    } else {
        return new PeriodicEvent(this->name().cast<std::string>(), _period.cast<T>(), nullptr, this->hide_mask());
    }
}

template<typename T>
std::vector<std::unique_ptr<Event<T>>> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, const py::iterable& args){
    if (events.is_none()){
        return {};
    }
    std::vector<std::unique_ptr<Event<T>>> res;
    for (py::handle item : events){
        Event<T>* ev_ptr = reinterpret_cast<Event<T>*>(item.cast<PyEvent&>().toEvent(shape, args));
        res.push_back(std::unique_ptr<Event<T>>(ev_ptr));
    }
    return res;
}

} // namespace ode

#endif // PY_EVENTS_IMPL_HPP