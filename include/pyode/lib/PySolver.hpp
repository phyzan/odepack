#ifndef PY_SOLVER_HPP
#define PY_SOLVER_HPP


#include "PyTools.hpp"
#include "../../ode/Core/VirtualBase.hpp"

namespace ode{

struct PyConstSolver : DtypeDispatcher {

    PyConstSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type);

    PyConstSolver(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}

    PyConstSolver(void* solver, PyStruct py_data, ScalarType scalar_type);

    PyConstSolver(const PyConstSolver& other);

    PyConstSolver(PyConstSolver&& other) noexcept;

    PyConstSolver& operator=(const PyConstSolver& other);

    PyConstSolver& operator=(PyConstSolver&& other) noexcept;

    virtual ~PyConstSolver();

    void set_pyobj(const PyConstSolver& other);

    template<typename T>
    OdeRichSolver<T>* cast();

    template<typename T>
    const OdeRichSolver<T>* cast() const;

    template<typename T>
    void init_solver(py::object f, py::object jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name);

    py::object          t0() const;

    py::object          q0() const;

    int                 direction() const; 

    py::object          t() const;

    py::object          t_last() const;

    py::object          t_old() const;

    py::object          q() const;

    py::object          q_last() const;

    py::object          q_old() const;

    py::object          stepsize() const;

    py::object          diverges() const;

    py::object          is_dead() const;

    py::object          Nsys() const;

    py::object          n_evals_rhs() const;

    py::object          n_evals_jac() const;

    void                show_state(int digits) const;

    py::object          py_rhs(const py::object& t, const py::iterable& py_q) const;

    py::object          py_jac(const py::object& t, const py::iterable& py_q) const;

    py::tuple           timeit_rhs(const py::object& t, const py::iterable& py_q) const;

    py::tuple           timeit_jac(const py::object& t, const py::iterable& py_q) const;

    bool                py_at_event(py::object event) const;

    py::str             status() const;    

    virtual py::object  copy() const;

    void* s = nullptr; //OdeRichSolver<T>*
    PyStruct data;
};


struct PySolver : public PyConstSolver {

    using PyConstSolver::PyConstSolver;

    PySolver(const PyConstSolver& other);

    DEFAULT_RULE_OF_FOUR(PySolver)

    py::object          advance();

    py::tuple           timeit_step();

    py::object          advance_to_event(const py::object& event);

    py::object          advance_until(const py::object& time, const py::object& observer, const py::object& extra_steps);

    void                reset();

    bool                set_ics(const py::object& t0, const py::iterable& py_q0, const py::object& dt, int direction);

    bool                resume();

    void                stop(const py::str& reason);

    void                kill(const py::str& reason);   
};

template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events);

// func::template operator()<T>(OdeRichSolver<T>* solver)
template<typename Callable>
void py_advance_all_general(py::object& list, Callable&& func, int threads, bool display_progress);


void py_advance_all(py::object& list, double t_goal, int threads, bool display_progress);

void py_advance_all_to_event(py::object& list, const py::str& event, double tmax, int threads, bool display_progress);

} // namespace ode


#endif // PY_SOLVER_HPP