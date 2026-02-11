#ifndef PY_SOLVER_HPP
#define PY_SOLVER_HPP


#include "PyTools.hpp"
#include "../../ode/Core/VirtualBase.hpp"

namespace ode{

struct PySolver : DtypeDispatcher {

    PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type);

    PySolver(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}

    PySolver(void* solver, PyStruct py_data, int scalar_type);

    PySolver(const PySolver& other);

    PySolver(PySolver&& other) noexcept;

    PySolver& operator=(const PySolver& other);

    PySolver& operator=(PySolver&& other) noexcept;

    virtual ~PySolver();

    void set_pyobj(const PySolver& other);

    template<typename T>
    OdeRichSolver<T>* cast();

    template<typename T>
    const OdeRichSolver<T>* cast() const;

    template<typename T>
    void init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name);


    py::object          t() const;

    py::object          t_old() const;

    py::object          q() const;

    py::object          q_old() const;

    py::object          stepsize() const;

    py::object          diverges() const;

    py::object          is_dead() const;

    py::object          Nsys() const;

    py::object          n_evals_rhs() const;

    void                show_state(int digits) const;

    py::object          py_rhs(const py::object& t, const py::iterable& py_q) const;

    py::object          py_jac(const py::object& t, const py::iterable& py_q) const;

    py::object          advance();

    py::object          advance_to_event();

    py::object          advance_until(const py::object& time);

    virtual py::object  copy() const = 0;

    void                reset();

    bool                set_ics(const py::object& t0, const py::iterable& py_q0, const py::object& dt, int direction);

    bool                resume();

    py::str             message() const;       

    py::object          py_at_event() const;

    py::object          py_event_located(const py::str& name) const;

    void* s = nullptr; //OdeRichSolver<T>*
    PyStruct data;
};

template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events);


void py_advance_all(py::object& list, double t_goal, int threads, bool display_progress);

} // namespace ode


#endif // PY_SOLVER_HPP