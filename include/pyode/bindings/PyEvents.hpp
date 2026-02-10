#ifndef PY_EVENTS_HPP
#define PY_EVENTS_HPP

#include "../pytools/pytools.hpp"

namespace ode{

class PyEvent : public DtypeDispatcher{

public:

    PyEvent(std::string name, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    template<typename T>
    Func<T>             mask() const;

    py::str             name() const;

    py::bool_           hide_mask() const;

    bool                is_lowlevel() const;

    void                check_sizes(size_t Nsys, size_t Nargs) const;

    virtual void*       toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) = 0;

    virtual ~PyEvent() = default;

protected:

    std::string _name;
    bool _hide_mask;
    void* _mask = nullptr;
    size_t _Nsys;
    size_t _Nargs;
    PyStruct data;
    bool _py_mask = false; //if true, cast _mask to py_mask<T>
};


class PyPrecEvent : public PyEvent {

public:

    PyPrecEvent(std::string name, py::object when, int dir, py::object mask, bool hide_mask, py::object event_tol, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPrecEvent);

    template<typename T>
    ObjFun<T> when() const;

    py::object event_tol() const;

    void* toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

    template<typename T>
    void* get_new_event();

protected:

    int _dir = 0;
    py::object _event_tol;
    void* _when = nullptr; //type: ObjFun<T>. If null, set it to py_event<T>
};



class PyPerEvent : public PyEvent{

public:

    PyPerEvent(std::string name, py::object period, py::object mask, bool hide_mask, const std::string& scalar_type, size_t Nsys, size_t Nargs);

    DEFAULT_RULE_OF_FOUR(PyPerEvent);

    template<typename T>
    void* get_new_event();

    void* toEvent(const std::vector<py::ssize_t>& shape, const py::tuple& args) override;

    py::object period() const;

private:
    py::object _period;

};

template<typename T>
std::vector<Event<T>*> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, const py::iterable& args);

bool all_are_lowlevel(const py::iterable& events);

} // namespace ode


#endif // PY_EVENTS_HPP