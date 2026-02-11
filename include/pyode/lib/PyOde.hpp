#ifndef PY_ODE_HPP
#define PY_ODE_HPP

#include "PyTools.hpp"

namespace ode{

class PyODE : public DtypeDispatcher{

public:

    PyODE(const py::object& f, const py::object& t0, const py::iterable& py_q0, const py::object& jacobian, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

    template<typename T, typename RhsType, typename JacType>
    PyODE(ODE_CONSTRUCTOR(T));

protected:

    PyODE(const std::string& scalar_type); //derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE& other);

    PyODE(PyODE&& other) noexcept;

    PyODE& operator=(const PyODE& other);

    PyODE& operator=(PyODE&& other) noexcept;

    virtual ~PyODE();

    template<typename T>
    ODE<T>* cast();

    template<typename T>
    const ODE<T>* cast() const;

    py::object call_Rhs(const py::object& t, const py::iterable& py_q) const;

    py::object call_Jac(const py::object& t, const py::iterable& py_q) const;

    py::object py_integrate(const py::object& interval, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object py_rich_integrate(const py::object& interval, const py::iterable& event_options, int max_prints);

    py::object py_integrate_until(const py::object& t, const py::object& t_eval, const py::iterable& event_options, int max_prints);

    py::object t_array() const;

    py::object q_array() const;

    py::tuple event_data(const py::str& event) const;

    virtual py::object copy() const;

    py::object solver_copy() const;

    py::dict event_map() const;

    py::object Nsys() const;

    py::object runtime() const;

    py::object diverges() const;

    py::object is_dead() const;

    void reset();

    void clear();

    void* ode = nullptr; //cast to ODE<T>*
    PyStruct data;
};


void py_integrate_all(py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress);

} // namespace ode

#endif // PY_ODE_HPP