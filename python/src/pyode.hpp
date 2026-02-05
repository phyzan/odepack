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
    const OdeRichSolver<T>* cast()const;

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
    inline const NormalizationEvent<T>& main_event() const;

    inline py::object py_logksi() const;

    inline py::object py_lyap() const;

    inline py::object py_t_lyap() const;

    inline py::object py_delta_s() const;

    py::object copy() const override;

};


struct PyRK23 : public PySolver{

    PyRK23(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK23(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    DEFAULT_RULE_OF_FOUR(PyRK23)

    py::object copy() const override;

};


struct PyRK45 : public PySolver{

    PyRK45(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK45(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    DEFAULT_RULE_OF_FOUR(PyRK45)

    py::object copy() const override;

};


struct PyDOP853 : public PySolver{

    PyDOP853(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyDOP853(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    DEFAULT_RULE_OF_FOUR(PyDOP853)

    py::object copy() const override;

};


struct PyBDF : public PySolver{

    PyBDF(const py::object& f, const py::object& t0, const py::iterable& q0, const py::object& jac, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyBDF(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}

    DEFAULT_RULE_OF_FOUR(PyBDF)

    py::object copy() const override;

};

struct PyRK4 : public PySolver{

    PyRK4(const py::object& ode, const py::object& t0, const py::iterable& q0, const py::object& rtol, const py::object& atol, const py::object& min_step, const py::object& max_step, const py::object& stepsize, int dir, const py::iterable& args, const py::iterable& events, const std::string& scalar_type);

    PyRK4(void* solver, PyStruct py_data, int scalar_type) : PySolver(solver, std::move(py_data), scalar_type) {}
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
    inline const OdeResult<T>*  cast() const;

    template<typename T>
    inline OdeResult<T>*        cast();

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

    PyODE(const std::string& scalar_type) : DtypeDispatcher(scalar_type){}//derived classes manage ode and q0_shape creation

public:

    PyODE(const PyODE& other);

    PyODE(PyODE&& other) noexcept;

    PyODE& operator=(const PyODE& other);

    PyODE& operator=(PyODE&& other) noexcept;

    virtual ~PyODE();

    template<typename T>
    inline ODE<T>* cast();

    template<typename T>
    inline const ODE<T>* cast() const;

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
class PyVecField : public SampledVectorField<double, NDIM> {

    using Base = SampledVectorField<double, NDIM>;

private:

    template<size_t... I, typename... PyArray>
    PyVecField(std::nullptr_t, std::index_sequence<I...>, const PyArray&... args) : Base(pack_elem<I>(args...)..., pack_elem<I+NDIM>(args...).data()...) {}

public:

    template<typename... PyArray>
    PyVecField(const PyArray&... args) : PyVecField(parse_args(args...), std::make_index_sequence<NDIM>(), args...) {}

    template<size_t Axis>
    inline py::object py_x() const{
        return py::cast(this->x(Axis));
    }

    template<size_t FieldIdx>
    inline py::object py_vx() const{
        return py::cast(this->field(FieldIdx));
    }

    template<size_t FieldIdx, typename... Scalar>
    inline py::object py_vx_at(Scalar... x) const{
        static_assert(sizeof...(Scalar) == NDIM, "Number of coordinates must match NDIM");
        this->check_coords(x...);
        std::array<double, NDIM> coords = {double(x)...};
        return py::cast(this->template get_single<FieldIdx>(coords.data()));
    }

    template<typename... Scalar>
    inline py::object py_vector(Scalar... x) const{
        static_assert(sizeof...(Scalar) == NDIM, "Number of coordinates must match NDIM");
        this->check_coords(x...);
        Array1D<double, NDIM> res;
        this->fill(res.data(), x...);
        return py::cast(res);
    }

    template<typename... Scalar>
    inline bool py_in_bounds(Scalar... x) const{
        return this->coords_in_bounds(x...);
    }

    // ==========================================

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const{

        if (q0.ndim() != 1 || q0.shape(0) != NDIM){
            throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(NDIM));
        }

        EXPAND(size_t, NDIM, I,
            check_coords(q0.at(I)...);
        );
    
        StepSequence<double> t_seq = to_step_sequence<double>(t_eval);
        try{
            double max_step_val = (max_step.is_none() ? inf<double>() : max_step.cast<double>());
            auto* result = new OdeResult<double>(this->streamline(q0.data(), length, rtol, atol, min_step, max_step_val, stepsize, direction, t_seq, method.cast<std::string>()));
            PyOdeResult py_res(result, {NDIM}, DTYPE_MAP.at("double"));
            return py::cast(py_res);
        } catch (const std::runtime_error& e){
            throw py::value_error(e.what());
        }
    }

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const{
        if (direction != 1 && direction != -1){
            throw py::value_error("Direction must be either 1 (forward) or -1 (backward)");
        }else if (q0.ndim() != 1 || q0.shape(0) != NDIM){
            throw py::value_error("Initial conditions must be a 1D array of length " + std::to_string(NDIM));
        }

        EXPAND(size_t, NDIM, I,
            check_coords(q0.at(I)...);
        );

        if (normalized){
            return py::cast(PyODE(OdeData{.rhs=this->ode_func_norm()}, 0., q0.data(), NDIM, rtol, atol, min_step, (max_step.is_none() ? inf<double>() : max_step.cast<double>()), stepsize, direction, {}, {}, method.cast<std::string>()));
        }else{
            return py::cast(PyODE(OdeData{.rhs=this->ode_func()}, 0., q0.data(), NDIM, rtol, atol, min_step, (max_step.is_none() ? inf<double>() : max_step.cast<double>()), stepsize, direction, {}, {}, method.cast<std::string>()));
        }
    }

    py::object py_streamplot_data(double max_length, double ds, int density) const{
        if (density <= 1){
            throw py::value_error("Density must be greater than 1");
        }
        if (max_length <= 0){
            throw py::value_error("Max length must be a positive number");
        }
        if (ds <= 0){
            throw py::value_error("ds must be a positive number");
        }

        std::vector<Array2D<double, NDIM, 0>> streamlines = this->streamplot_data(max_length, ds, size_t(density));
        py::list result;
        for (const Array2D<double, NDIM, 0>& line : streamlines){
            result.append(py::cast(line));
        }
        return result;
    }

private:

    template<typename... Scalar>
    inline void check_coords(Scalar... x) const{
        static_assert(sizeof...(Scalar) == NDIM, "Number of coordinates must match NDIM");
        if (!this->coords_in_bounds(x...)){
            py::array_t<double> py_coords = py::cast(Array1D<double, NDIM>{double(x)...});
            throw py::value_error("Coordinates " + py::repr(py_coords).cast<std::string>() + " are out of bounds");
        }
    }

    template<typename... PyArray>
    static std::nullptr_t parse_args(const PyArray&... args){
        EXPAND(size_t, NDIM, I,
            if (((pack_elem<I>(args...).ndim() != 1) || ...)){
                throw py::value_error("All grid dimensions must be 1D arrays");
            }else if (((pack_elem<I+NDIM>(args...).ndim() != NDIM) || ...)){
                throw py::value_error("All vector component arrays must be " + std::to_string(NDIM) + "D");
            }else if (((!is_sorted(pack_elem<I>(args...))) || ...)){
                throw py::value_error("All grid dimensions must be 1D arrays");
            }
            FOR_LOOP(size_t, J, NDIM,
                const py::array_t<double>& field = pack_elem<J+NDIM>(args...);
                if (((field.shape(I) != pack_elem<I>(args...).size()) || ...)){
                    throw py::value_error("Grid size does not match vector component array size");
                }
            );
        );
        return nullptr;
    }

    static bool is_sorted(const py::array_t<double>& arr){
        const double* ptr = arr.data();
        for (ssize_t i = 1; i < arr.size(); ++i){
            if (ptr[i] <= ptr[i-1]){
                return false;
            }
        }
        return true;
    }

};


class PyVecFieldBase {

public:

    virtual ~PyVecFieldBase() = default;

    virtual py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const = 0;

    virtual py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const = 0;

    virtual py::object py_streamplot_data(double max_length, double ds, int density) const = 0;

};


class PyVecField2D : public PyVecFieldBase, public PyVecField<2> {
    
    using Base = PyVecField<2>;

public:

    PyVecField2D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& vx, const py::array_t<double>& vy) : Base(x, y, vx, vy) {}

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const override{
        return Base::py_streamline(q0, length, rtol, atol, min_step, max_step, stepsize, direction, t_eval, method);
    }

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const override{
        return Base::py_streamline_ode(q0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    }

    py::object py_streamplot_data(double max_length, double ds, int density) const override{
        return Base::py_streamplot_data(max_length, ds, density);
    }

};

class PyVecField3D : public PyVecFieldBase, public PyVecField<3> {
    
    using Base = PyVecField<3>;

public:

    PyVecField3D(const py::array_t<double>& x, const py::array_t<double>& y, const py::array_t<double>& z, const py::array_t<double>& vx, const py::array_t<double>& vy, const py::array_t<double>& vz) : Base(x, y, z, vx, vy, vz) {}

    py::object py_streamline(const py::array_t<double>& q0, double length, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::object& t_eval, const py::str& method) const override{
        return Base::py_streamline(q0, length, rtol, atol, min_step, max_step, stepsize, direction, t_eval, method);
    }

    py::object py_streamline_ode(const py::array_t<double>& q0, double rtol, double atol, double min_step, const py::object& max_step, double stepsize, int direction, const py::str& method, bool normalized) const override{
        return Base::py_streamline_ode(q0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    }

    py::object py_streamplot_data(double max_length, double ds, int density) const override{
        return Base::py_streamplot_data(max_length, ds, density);
    }
    
};

void py_integrate_all(py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress);

void py_advance_all(py::object& list, double t_goal, int threads, bool display_progress);

bool all_are_lowlevel(const py::iterable& events);

template<typename T>
std::string get_scalar_type();

template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events);

/*
IMPLEMENTATIONS
*/




//===========================================================================================
//                          Template Method Implementations
//===========================================================================================


template<typename T>
OdeRichSolver<T>* PySolver::cast(){
    return reinterpret_cast<OdeRichSolver<T>*>(this->s);
}

template<typename T>
const OdeRichSolver<T>* PySolver::cast()const{
    return reinterpret_cast<const OdeRichSolver<T>*>(this->s);
}


template<typename T>
inline const NormalizationEvent<T>& PyVarSolver::main_event() const{
    return static_cast<const NormalizationEvent<T>&>(reinterpret_cast<const OdeRichSolver<T>*>(this->s)->event_col().event(0));
}

template<typename T>
inline const OdeResult<T>* PyOdeResult::cast() const{
    return reinterpret_cast<OdeResult<T>*>(this->res);
}

template<typename T>
inline OdeResult<T>* PyOdeResult::cast() {
    return reinterpret_cast<OdeResult<T>*>(this->res);
}


template<typename T>
Func<T> PyEvent::mask() const{
    if (_py_mask){
        return py_mask<T>;
    }
    else if (this->_mask != nullptr){
        return reinterpret_cast<Func<T>>(this->_mask);
    }
    else{
        return nullptr;
    }
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
    return new ObjectOwningEvent<PreciseEvent<T>, PyStruct>(this->data, this->name(), this->when<T>(), _dir, this->mask<T>(), this->hide_mask(), this->_event_tol.cast<T>());
}

template<typename T>
void* PyPerEvent::get_new_event(){
    return new ObjectOwningEvent<PeriodicEvent<T>, PyStruct>(this->data, this->name(), _period.cast<T>(), this->mask<T>(), this->hide_mask());
}

template<typename T>
std::vector<Event<T>*> to_Events(const py::iterable& events, const std::vector<py::ssize_t>& shape, const py::iterable& args){
    if (events.is_none()){
        return {};
    }
    std::vector<Event<T>*> res;
    for (py::handle item : events){
        Event<T>* ev_ptr = reinterpret_cast<Event<T>*>(item.cast<PyEvent&>().toEvent(shape, args));
        res.push_back(ev_ptr);
    }
    return res;
}


template<typename T>
inline ODE<T>* PyODE::cast(){
    return reinterpret_cast<ODE<T>*>(this->ode);
}

template<typename T>
inline const ODE<T>* PyODE::cast() const {
    return reinterpret_cast<const ODE<T>*>(this->ode);
}




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
py::object PyOdeSolution::_get_frame(const py::object& t) const{
    return py::cast(Array<T>(reinterpret_cast<OdeSolution<T>*>(this->res)->operator()(t.cast<T>()).data(), this->q0_shape.data(), this->q0_shape.size()));
}

template<typename T>
py::object PyOdeSolution::_get_array(const py::array& py_array) const{
    const auto nt = size_t(py_array.size());
    std::vector<py::ssize_t> final_shape(py_array.shape(), py_array.shape()+py_array.ndim());
    final_shape.insert(final_shape.end(), this->q0_shape.begin(), this->q0_shape.end());
    Array<T> res(final_shape.data(), final_shape.size());
    const auto* solution = reinterpret_cast<const OdeSolution<T>*>(this->res);

    // Extract array values and cast them to T using Python's item access
    for (size_t i=0; i<nt; i++){
        py::object item = py_array.attr("flat")[py::int_(i)];
        T t_value = py::cast<T>(item);
        copy_array(res.data()+i*nsys, solution->operator()(t_value).data(), nsys);
    }
    return py::cast(res);
}


template<typename T>
VariationalODE<T, 0, Func<T>, Func<T>>& PyVarODE::varode(){
    return *static_cast<VariationalODE<T, 0, Func<T>, Func<T>>*>(this->ode);
}

template<typename T>
const VariationalODE<T, 0, Func<T>, Func<T>>& PyVarODE::varode() const {
    return *static_cast<const VariationalODE<T, 0, Func<T>, Func<T>>*>(this->ode);
}


//===========================================================================================
//                                      Templated functions
//===========================================================================================

template<typename T>
OdeData<Func<T>, void> init_ode_data(PyStruct& data, std::vector<T>& args, const py::object& f, const py::iterable& q0, const py::object& jacobian, const py::iterable& py_args, const py::iterable& events){
    std::string scalar_type = get_scalar_type<T>();
    data.shape = shape(q0);
    data.py_args = py::tuple(py_args);
    size_t _size = prod(data.shape);

    bool f_is_compiled = py::isinstance<PyFuncWrapper>(f) || py::isinstance<py::capsule>(f);
    bool jac_is_compiled = !jacobian.is_none() && (py::isinstance<PyFuncWrapper>(jacobian) || py::isinstance<py::capsule>(jacobian));
    args = (f_is_compiled || jac_is_compiled ? toCPP_Array<T, std::vector<T>>(py_args) : std::vector<T>{});
    OdeData<Func<T>, void> ode_rhs = {nullptr, nullptr, &data};
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
            throw py::value_error("All objects in 'events' iterable argument must be instances of the Event class, not " + py::str(ev.get_type()).cast<std::string>());
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
    return ode_rhs;
}

template<typename T>
std::string get_scalar_type(){
    if constexpr (std::is_same_v<T, float>){
        return "float";
    }
    else if constexpr (std::is_same_v<T, double>){
        return "double";
    }
    else if constexpr (std::is_same_v<T, long double>){
        return "long double";
    }
#ifdef MPREAL
    else if constexpr (std::is_same_v<T, mpfr::mpreal>){
        return "mpreal";
    }
#endif
    else{
        static_assert(false, "Unsupported scalar type T");
    }
}

} // namespace ode

#endif
