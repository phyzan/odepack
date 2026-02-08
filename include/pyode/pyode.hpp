#ifndef PYODE_HPP
#define PYODE_HPP

#include "pytools.hpp"


namespace ode{

struct DtypeDispatcher{

    DtypeDispatcher(const std::string& dtype_);

    DtypeDispatcher(int dtype_);

    int scalar_type;
};

struct PyFuncWrapper : DtypeDispatcher {

    void* rhs; //Func<T>
    size_t Nsys;
    std::vector<py::ssize_t> output_shape;
    size_t Nargs;
    size_t output_size;

    PyFuncWrapper(const py::capsule& obj, py::ssize_t Nsys, const py::array_t<py::ssize_t>& output_shape, py::ssize_t Nargs, const std::string& scalar_type);

    template<typename T>
    py::object call_impl(const py::object& t, const py::iterable& py_q, py::args py_args) const;

    py::object call(const py::object& t, const py::iterable& py_q, const py::args& py_args) const;
};


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


struct PySolver : DtypeDispatcher {

    PySolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& py_events, const std::string& name, const std::string& scalar_type);

    PySolver(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}

    PySolver(void* solver, PyStruct py_data, int scalar_type);

    PySolver(const PySolver& other);

    PySolver(PySolver&& other) noexcept;

    PySolver& operator=(const PySolver& other);

    PySolver& operator=(PySolver&& other) noexcept;

    virtual ~PySolver();

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


struct PyVarSolver : public PySolver{

    PyVarSolver(const py::object& f, const py::object& jac, const py::object& t0, const py::iterable& py_q0, const py::object& period, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const std::string& method, const std::string& scalar_type);

    PyVarSolver(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarSolver)

    template<typename T>
    const NormalizationEvent<T>& main_event() const;

    py::object py_logksi() const;

    py::object py_lyap() const;

    py::object py_t_lyap() const;

    py::object py_delta_s() const;

    py::object copy() const override;

};


struct PyRK23 : public PySolver{

    PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK23(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK23)

    py::object copy() const override;

};


struct PyRK45 : public PySolver{

    PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK45(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK45)

    py::object copy() const override;

};


struct PyDOP853 : public PySolver{

    PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyDOP853(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyDOP853)

    py::object copy() const override;

};


struct PyBDF : public PySolver{

    PyBDF(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyBDF(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyBDF)

    py::object copy() const override;

};

struct PyRK4 : public PySolver{

    PyRK4(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK4(void* solver, PyStruct py_data, int scalar_type);

    DEFAULT_RULE_OF_FOUR(PyRK4)

    py::object copy() const override;

};


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



class PyVarODE : public PyODE{

public:

    PyVarODE(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& period, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& py_args, const py::iterable& events, const py::str& method, const std::string& scalar_type);

    DEFAULT_RULE_OF_FOUR(PyVarODE);

    template<typename T>
    VariationalODE<T, 0, Func<T>, Func<T>>& varode();

    template<typename T>
    const VariationalODE<T, 0, Func<T>, Func<T>>& varode() const;

    py::object py_t_lyap() const;

    py::object py_lyap() const;

    py::object py_kicks() const;

    py::object copy() const override;
    
};


template<size_t NDIM>
class PyScalarField : public RegularGridInterpolator<double, NDIM> {

    using Base = RegularGridInterpolator<double, NDIM>;

private:

    template<size_t... I, typename... PyArray>
    PyScalarField(std::nullptr_t, std::index_sequence<I...>, const PyArray&... args);

public:

    template<typename... PyArray>
    PyScalarField(const PyArray&... args);

    template<typename... Scalar>
    double operator()(Scalar... x) const;

private:

    template<typename... PyArray>
    static std::nullptr_t parse_args(const PyArray&... args);
};


class PyScatteredField : public LinearNdInterpolator<double> {

    using Base = LinearNdInterpolator<double>;

public:

    // Python signature is ScatteredField(x: np.ndarray (npoints, ndim), values: np.ndarray (npoints)), where
    PyScatteredField(const py::array_t<double>& x, const py::array_t<double>& values);

    template<typename... Scalar>
    double operator()(Scalar... x) const;

    py::object py_points() const;

    py::object py_values() const;

    double py_value_at(const py::args& x) const;

private:

    PyScatteredField(std::nullptr_t, const py::array_t<double>& x, const py::array_t<double>& values);

    static std::nullptr_t parse_args(const py::array_t<double>& x, const py::array_t<double>& values);

};


class PyVecFieldBase {

public:

    virtual ~PyVecFieldBase() = default;

    virtual py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const = 0;

    virtual py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const = 0;

    virtual py::object py_streamplot_data(double max_length, double ds, int density) const = 0;

};


template<size_t NDIM>
class PyVecField : public SampledVectorField<double, NDIM>, public PyVecFieldBase {

    using Base = SampledVectorField<double, NDIM>;

private:

    template<size_t... I, typename... PyArray>
    PyVecField(std::nullptr_t, std::index_sequence<I...>, const PyArray&... args);

public:

    template<typename... PyArray>
    PyVecField(const PyArray&... args);

    template<size_t Axis>
    py::object py_x() const;

    template<size_t FieldIdx>
    py::object py_vx() const;

    template<size_t FieldIdx, typename... Scalar>
    py::object py_vx_at(Scalar... x) const;

    template<typename... Scalar>
    py::object py_vector(Scalar... x) const;

    template<typename... Scalar>
    bool py_in_bounds(Scalar... x) const;

    // ==========================================

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const override;

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const override;

    py::object py_streamplot_data(double max_length, double ds, int density) const override;

private:

    template<typename... Scalar>
    void check_coords(Scalar... x) const;

    template<typename... PyArray>
    static std::nullptr_t parse_args(const PyArray&... args);



};


class PyVecField2D : public PyVecField<2> {
    
    using Base = PyVecField<2>;

public:

    PyVecField2D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& vx, const py::array_t<double>& vy);

    using Base::py_streamline, Base::py_streamline_ode, Base::py_streamplot_data;

};

class PyVecField3D : public PyVecField<3> {
    
    using Base = PyVecField<3>;

public:

    PyVecField3D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& z, const py::array_t<double>& vx, const py::array_t<double>& vy, const py::array_t<double>& vz);

    using Base::py_streamline, Base::py_streamline_ode, Base::py_streamplot_data;
    
};

template<typename T>
std::vector<Event<T>*> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, const py::iterable& args);

bool is_sorted(const py::array_t<double>& arr);

void py_integrate_all(py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress);

void py_advance_all(py::object& list, double t_goal, int threads, bool display_progress);

bool all_are_lowlevel(const py::iterable& events);

template<typename T>
std::string get_scalar_type();

template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events);


} // namespace ode

#endif //PYODE_SOLVERS_HPP
