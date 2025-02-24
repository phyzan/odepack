#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include <ode.hpp>


namespace py = pybind11;


template<class Tt, size_t N>
ode_f<Tt, N> to_Ode(const py::object& f);

template<class Tt, size_t N>
event_f<Tt, N> to_event(const py::object py_event);

template<class ArrayType, class T>
ArrayType toCPP_Array(const py::array& A);

template<class T, class ArrayType>
py::array_t<T> to_numpy(const ArrayType& array, const std::vector<size_t>& shape);

template<class Tt, size_t N>
std::vector<Tt> flatten(const std::vector<vec<Tt, N>>&);

template<class T>
py::tuple to_tuple(const std::vector<T>& vec);


#pragma GCC visibility push(hidden)
template<class T>
struct PyOdeResult{
    std::vector<T> t_src;
    std::vector<T> y_src;
    py::array_t<T> t;
    py::array_t<T> y;
    bool diverges;
    bool is_stiff;
    long double runtime;
};


template<class T>
struct PyOdeArgs{

    py::tuple ics;
    T t;
    T h;
    T rtol;
    T atol;
    T cutoff_step;
    py::str method;
    size_t max_frames;
    py::tuple pyargs;
    py::object getcond;
    py::object breakcond;

};


template<class Tt, size_t N = 0>
class PyOde : public ODE<Tt, N, false, false>{

    public:
        PyOde(const py::object& df): ODE<Tt, N, false, false>(to_Ode<Tt, N>(df)) {}

        PyOde(ode_f<Tt, N> df): ODE<Tt, N, false, false>(df) {}

        PyOde<Tt, N> copy() const;
        
        const PyOdeResult<Tt> pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const;
        
        py::list pysolve_all(const py::list&, int threads) const;

        static py::list py_dsolve_all(const py::list& data, int threads);

};


template<class Tt, size_t N>
struct PyOdeSet{

    PyOde<Tt, N> ode;
    PyOdeArgs<Tt> params;

};
#pragma GCC visibility pop


template<class Tt, size_t N>
OdeArgs<Tt, N, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams);

template<class Tt, size_t N>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, N>& res);



/*
------------------------------------------------------------------------------------
-------------------------- IMPLEMENTATIONS -----------------------------------------
------------------------------------------------------------------------------------
*/

template<class Tt, size_t N>
ode_f<Tt, N> to_Ode(const py::object& f){
    ode_f<Tt, N> g = [f](const Tt& t, const vec<Tt, N>& y, const std::vector<Tt>& args) -> vec<Tt, N> {
        py::array_t<Tt> _y  = to_numpy<Tt>(y, {static_cast<size_t>(y.size())});
        py::tuple pyargs = to_tuple(args);
        return toCPP_Array<vec<Tt, N>, Tt>(f(t, _y, pyargs));
    };
    return g;
}

template<class Tt, size_t N>
event_f<Tt, N> to_event(const py::object py_event){
    event_f<Tt, N> g = [py_event](const Tt& t1, const vec<Tt, N>& f1, const Tt& t2, const vec<Tt, N>& f2) -> bool {
        bool res = py_event(t1, t2, to_numpy<Tt>(f1, {static_cast<size_t>(f1.size())}), to_numpy<Tt>(f2, {static_cast<size_t>(f2.size())})).equal(py::bool_(true));
        return res;
    };
    return g;
}

template<class ArrayType, class T>
ArrayType toCPP_Array(const py::array& A){
    size_t n = A.size();
    ArrayType res(n);

    const T* data = static_cast<const T*>(A.data());

    for (size_t i=0; i<n; i++){
        // res[i] = data[i].template cast<T>();
        res[i] = data[i];
    }
    return res;
}

template<class T, class ArrayType>
py::array_t<T> to_numpy(const ArrayType& array, const std::vector<size_t>& shape){
    return py::array_t<T>(shape, array.data());
}

template<class Tt, size_t N>
std::vector<Tt> flatten(const std::vector<vec<Tt, N>>& f){
    size_t nt = f.size();
    size_t nd = f[0].size();
    std::vector<Tt> res(nt*nd);

    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<nd; j++){
            res[i*nd + j] = f[i][j];
        }
    }
    return res;
}

template<class T>
py::tuple to_tuple(const std::vector<T>& arr) {
    py::tuple py_tuple(arr.size());  // Create a tuple of the same size as the vector
    for (size_t i = 0; i < arr.size(); ++i) {
        py_tuple[i] = py::float_(arr[i]);  // Convert each double element to py::float_
    }
    return py_tuple;
}


template<class Tt, size_t N>
OdeArgs<Tt, N, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams){
    ICS<Tt, N> ics = {pyparams.ics[0].template cast<Tt>(), toCPP_Array<vec<Tt, N>, Tt>(pyparams.ics[1])};
    Tt t = pyparams.t;
    Tt h = pyparams.h;
    Tt rtol = pyparams.rtol;
    Tt atol = pyparams.atol;
    Tt cutoff_step = pyparams.cutoff_step;
    std::string method = pyparams.method.template cast<std::string>();
    size_t max_frames = pyparams.max_frames;
    std::vector<Tt> args;
    event_f<Tt, N> getcond = nullptr;
    event_f<Tt, N> breakcond = nullptr;

    if (!pyparams.pyargs.empty()){
        args = toCPP_Array<std::vector<Tt>, Tt>(pyparams.pyargs);
    }

    if (!pyparams.getcond.is(py::none())) {
        getcond = to_event<Tt, N>(pyparams.getcond);
    }
    
    if (!pyparams.breakcond.is(py::none())) {
        breakcond = to_event<Tt, N>(pyparams.breakcond);
    }

    OdeArgs<Tt, N, false> res = {ics, t, h, rtol, atol, cutoff_step, method, max_frames, args, getcond, breakcond};
    return res;
}


template<class Tt, size_t N>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, N>& res){

    std::vector<Tt> f_flat = flatten<Tt, N>(res.y);
    size_t nd = res.y[0].size();
    size_t nt = res.y.size();

    PyOdeResult<Tt> pyres{res.t, f_flat, to_numpy<Tt>(res.t, {nt}), to_numpy<Tt>(f_flat, {nt, nd}), res.diverges, res.is_stiff, res.runtime};

    return pyres;
}


/*
----------------------CLASS IMPLEMENTATION---------------------
*/


template<class Tt, size_t N>
const PyOdeResult<Tt> PyOde<Tt, N>::pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const {

    const PyOdeArgs<Tt> pyparams = {ics, t, dt, rtol, atol, cutoff_step, method, max_frames, pyargs, getcond, breakcond};
    OdeArgs<Tt, N, false> ode_args = to_OdeArgs<Tt, N>(pyparams);

    OdeResult<Tt, N> res = ODE<Tt, N, false, false>::solve(ode_args);

    PyOdeResult<Tt> pyres = to_PyOdeResult(res);

    return pyres;
}

template<class Tt, size_t N>
py::list PyOde<Tt, N>::pysolve_all(const py::list& pyparams, int threads) const{

    py::list data;
    size_t n = pyparams.size();
    for (size_t i = 0; i < n; i++){
        data.append(py::make_tuple(*this, pyparams[i]));
    }
    
    py::list res = py_dsolve_all(data, threads);

    return res;
}

template<class Tt, size_t N>
PyOde<Tt, N> PyOde<Tt, N>::copy() const{
    PyOde<Tt, N> res(this->f);
    return res;
}


template<class Tt, size_t N>
py::list PyOde<Tt, N>::py_dsolve_all(const py::list& data, int threads){
    
    // if (...){
    //     throw std::runtime_error("Cannot integrate many ode's using multithreading when the ode is given in python syntax, duo to GIL limitations");
    // }

    size_t n = data.size();
    std::vector<OdeSet<Tt, N, false, false>> odeset(n);
    std::vector<OdeResult<Tt, N>> ode_res;
    py::list res;

    //cast py_params to cpp params
    py::tuple tup;
    py::dict kw;
    py::list pyparams;
    for (size_t i=0; i<n; i++){
        tup = data[i].cast<py::tuple>();
        kw = tup[1].cast<py::dict>();
        
        PyOdeArgs<Tt> pystruct = {kw["ics"], kw["t"].cast<Tt>(), kw["dt"].cast<Tt>(), kw["rtol"].cast<Tt>(), kw["atol"].cast<Tt>(), kw["cutoff_step"].cast<Tt>(), kw["method"], kw["max_frames"].cast<size_t>(), kw["args"], py::none(), py::none()};
        odeset[i] = { tup[0].cast<PyOde<Tt, N>>(), to_OdeArgs<Tt, N>(pystruct)};
        if (kw.size() > 9){
            throw std::runtime_error("When solving an ode in parallel, no more than 7 arguments can be passed in the ode, since the rest of them would be cast into python functions.GIL prevents the program to call python function in parallel");
        }
    }

    //retrieve array of results from base class method
    ode_res = dsolve_all(odeset, threads);
    //convert results to python type
    for (size_t i=0; i<n; i++){
        res.append(to_PyOdeResult(ode_res[i]));
    }
    
    return res;
}


template<class Tt, size_t N>
void define_ode_module(py::module& m) {
    py::class_<PyOde<Tt, N>>(m, "LowLevelODE", py::module_local())
        .def(py::init<py::object>(), py::arg("f"))
        .def("solve", &PyOde<Tt, N>::pysolve,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("rtol") = 1e-6,
            py::arg("atol") = 1e-12,
            py::arg("cutoff_step") = 0.,
            py::arg("method") = py::str("RK45"),
            py::arg("max_frames") = 0,
            py::arg("args") = py::tuple(),
            py::arg("getcond") = py::none(),
            py::arg("breakcond") = py::none())
        .def("solve_all", &PyOde<Tt, N>::pysolve_all,
            py::arg("params"),
            py::arg("threads") = -1)
        .def("copy", &PyOde<Tt, N>::copy)
        .def_static("dsolve_all", &PyOde<Tt, N>::py_dsolve_all,
            py::arg("data"),
            py::arg("threads") = -1)
        .def("__deepcopy__", [](const PyOde<Tt, N> &self, py::dict) {
            return self.copy();  // Calls copy constructor and returns a new object
        });


    py::class_<PyOdeResult<Tt>>(m, "OdeResult", py::module_local())
        .def_readonly("t", &PyOdeResult<Tt>::t)
        .def_readonly("y", &PyOdeResult<Tt>::y)
        .def_readonly("diverges", &PyOdeResult<Tt>::diverges)
        .def_readonly("is_stiff", &PyOdeResult<Tt>::is_stiff)
        .def_readonly("runtime", &PyOdeResult<Tt>::runtime);
}

template<class Tt, size_t N>
void define_lowlevel_ode(py::module& m, ode_f<Tt, N> func_ptr){
    define_ode_module<Tt, N>(m);

    m.def("ode", [func_ptr]() {
        return PyOde<Tt, N>(func_ptr);
    });
}

#endif