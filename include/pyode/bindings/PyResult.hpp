#ifndef PY_RESULT_HPP
#define PY_RESULT_HPP


#include "../pytools/pytools.hpp"

namespace ode{


struct PyOdeResult : DtypeDispatcher{

    PyOdeResult(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type);

    PyOdeResult(const PyOdeResult& other);

    PyOdeResult(PyOdeResult&& other) noexcept;

    PyOdeResult& operator=(const PyOdeResult& other);

    PyOdeResult& operator=(PyOdeResult&& other) noexcept;

    virtual ~PyOdeResult();

    template<typename T>
    const OdeResult<T>*  cast() const;

    template<typename T>
    OdeResult<T>*        cast();

    py::object                  t() const;

    py::object                  q() const;

    py::dict                    event_map() const;

    py::tuple                   event_data(const py::str& event) const;

    py::bool_                   diverges() const;

    py::bool_                   success() const;

    py::float_                  runtime() const;

    py::str                     message() const;

    void                        examine() const;

    void* res = nullptr;
    std::vector<py::ssize_t> q0_shape;

};


struct PyOdeSolution : public PyOdeResult{

    PyOdeSolution(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyOdeSolution);

    py::object operator()(const py::object& t) const;

    template<typename T>
    py::object _get_frame(const py::object& t) const;

    template<typename T>
    py::object _get_array(const py::array& py_array) const;

    size_t nsys;

};


}

#endif // PY_RESULT_HPP