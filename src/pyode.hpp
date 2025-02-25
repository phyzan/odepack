#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include "ode.hpp"


namespace py = pybind11;

template<class Tt, class Ty>
ode_f<Tt, Ty> to_ODE_callable(py::object& f);

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(const py::object py_event);

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A);


template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<size_t>& _shape = {});

template<class Tt, class Ty>
std::vector<Tt> flatten(const std::vector<Ty>&);

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


template<class Tt, class Ty>
class PyOde : public ODE<Tt, Ty, false, false>{

public:

    PyOde(ode_f<Tt, Ty> df): ODE<Tt, Ty, false, false>(df) {}

    PyOde(py::object df): ODE<Tt, Ty, false, false>(to_ODE_callable<Tt, Ty>(df)) {}

    PyOde<Tt, Ty> copy() const;
    
    const PyOdeResult<Tt> pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const;
    
    py::list pysolve_all(const py::list&, int threads) const;

    static py::list py_dsolve_all(const py::list& data, int threads);

};


template<class Tt, class Ty>
struct PyOdeSet{

    PyOde<Tt, Ty> ode;
    PyOdeArgs<Tt> params;

};
#pragma GCC visibility pop


template<class Tt, class Ty>
OdeArgs<Tt, Ty, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams);

template<class Tt, class Ty>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, Ty>& res);



/*
------------------------------------------------------------------------------------
-------------------------- IMPLEMENTATIONS -----------------------------------------
------------------------------------------------------------------------------------
*/

template<class Tt, class Ty>
ode_f<Tt, Ty> to_ODE_callable(py::object& f){
    ode_f<Tt, Ty> g = [f](const Tt& t, const Ty& y, const std::vector<Tt>& args) -> Ty {
        return toCPP_Array<Tt, Ty>(f(t, to_numpy<Tt>(y), *to_tuple(args)));
    };
    return g;
}

template<class Tt, class Ty>
event_f<Tt, Ty> to_event(const py::object py_event){
    event_f<Tt, Ty> g = [py_event](const Tt& t1, const Ty& f1, const Tt& t2, const Ty& f2) -> bool {
        bool res = py_event(t1, t2, to_numpy<Tt>(f1), to_numpy<Tt>(f2)).equal(py::bool_(true));
        return res;
    };
    return g;
}

template<class Tt, class Ty>
Ty toCPP_Array(const py::array& A){
    size_t n = A.size();
    Ty res(n);

    const Tt* data = static_cast<const Tt*>(A.data());

    for (size_t i=0; i<n; i++){
        res[i] = data[i];
    }
    return res;
}

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<size_t>& _shape){
    if (_shape.size() == 0){
        return py::array_t<Scalar>(shape(array), array.data());
    }
    else{
        return py::array_t<Scalar>(_shape, array.data());
    }
}

template<class Tt, class Ty>
std::vector<Tt> flatten(const std::vector<Ty>& f){
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


template<class Tt, class Ty>
OdeArgs<Tt, Ty, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams){
    ICS<Tt, Ty> ics = {pyparams.ics[0].template cast<Tt>(), toCPP_Array<Tt, Ty>(pyparams.ics[1])};
    Tt t = pyparams.t;
    Tt h = pyparams.h;
    Tt rtol = pyparams.rtol;
    Tt atol = pyparams.atol;
    Tt cutoff_step = pyparams.cutoff_step;
    std::string method = pyparams.method.template cast<std::string>();
    size_t max_frames = pyparams.max_frames;
    std::vector<Tt> args;
    event_f<Tt, Ty> getcond = nullptr;
    event_f<Tt, Ty> breakcond = nullptr;

    //CONVERT PYARGS TO POINTER FOR ARGS
    if (!pyparams.pyargs.empty()){
        args = toCPP_Array<Tt, std::vector<Tt>>(pyparams.pyargs);
    }

    if (!pyparams.getcond.is(py::none())) {
        getcond = to_event<Tt, Ty>(pyparams.getcond);
    }
    
    if (!pyparams.breakcond.is(py::none())) {
        breakcond = to_event<Tt, Ty>(pyparams.breakcond);
    }

    OdeArgs<Tt, Ty, false> res = {ics, t, h, rtol, atol, cutoff_step, method, max_frames, args, getcond, breakcond};
    return res;
}


template<class Tt, class Ty>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, Ty>& res){

    std::vector<Tt> f_flat = flatten<Tt, Ty>(res.y);
    size_t nd = res.y[0].size();
    size_t nt = res.y.size();

    PyOdeResult<Tt> pyres{res.t, f_flat, to_numpy<Tt>(res.t), to_numpy<Tt>(f_flat, {nt, nd}), res.diverges, res.is_stiff, res.runtime};

    return pyres;
}


/*
----------------------CLASS IMPLEMENTATION---------------------
*/


template<class Tt, class Ty>
const PyOdeResult<Tt> PyOde<Tt, Ty>::pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const {

    const PyOdeArgs<Tt> pyparams = {ics, t, dt, rtol, atol, cutoff_step, method, max_frames, pyargs, getcond, breakcond};
    OdeArgs<Tt, Ty, false> ode_args = to_OdeArgs<Tt, Ty>(pyparams);

    OdeResult<Tt, Ty> res = ODE<Tt, Ty, false, false>::solve(ode_args);

    PyOdeResult<Tt> pyres = to_PyOdeResult(res);

    return pyres;
}

template<class Tt, class Ty>
py::list PyOde<Tt, Ty>::pysolve_all(const py::list& pyparams, int threads) const{

    py::list data;
    size_t n = pyparams.size();
    for (size_t i = 0; i < n; i++){
        data.append(py::make_tuple(*this, pyparams[i]));
    }
    
    py::list res = py_dsolve_all(data, threads);

    return res;
}

template<class Tt, class Ty>
PyOde<Tt, Ty> PyOde<Tt, Ty>::copy() const{
    PyOde<Tt, Ty> res(this->f);
    return res;
}


template<class Tt, class Ty>
py::list PyOde<Tt, Ty>::py_dsolve_all(const py::list& data, int threads){
    
    // if (...){
    //     throw std::runtime_error("Cannot integrate many ode's using multithreading when the ode is given in python syntax, duo to GIL limitations");
    // }

    size_t n = data.size();
    std::vector<OdeSet<Tt, Ty, false, false>> odeset(n);
    std::vector<OdeResult<Tt, Ty>> ode_res;
    py::list res;

    //cast py_params to cpp params
    py::tuple tup;
    py::dict kw;
    py::list pyparams;
    for (size_t i=0; i<n; i++){
        tup = data[i].cast<py::tuple>();
        kw = tup[1].cast<py::dict>();
        
        PyOdeArgs<Tt> pystruct = {kw["ics"], kw["t"].cast<Tt>(), kw["dt"].cast<Tt>(), kw["rtol"].cast<Tt>(), kw["atol"].cast<Tt>(), kw["cutoff_step"].cast<Tt>(), kw["method"], kw["max_frames"].cast<size_t>(), kw["args"], py::none(), py::none()};
        odeset[i] = { tup[0].cast<PyOde<Tt, Ty>>(), to_OdeArgs<Tt, Ty>(pystruct)};
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


template<class Tt, class Ty>
void define_ode_module(py::module& m) {
    py::class_<PyOde<Tt, Ty>>(m, "LowLevelODE", py::module_local())
        .def(py::init<py::object>(), py::arg("f"))
        .def("solve", &PyOde<Tt, Ty>::pysolve,
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
        .def("solve_all", &PyOde<Tt, Ty>::pysolve_all,
            py::arg("parameter_list"),
            py::arg("threads") = -1)
        .def("copy", &PyOde<Tt, Ty>::copy)
        .def_static("dsolve_all", &PyOde<Tt, Ty>::py_dsolve_all,
            py::arg("data"),
            py::arg("threads") = -1)
        .def("__deepcopy__", [](const PyOde<Tt, Ty> &self, py::dict) {
            return self.copy();  // Calls copy constructor and returns a new object
        });


    py::class_<PyOdeResult<Tt>>(m, "OdeResult", py::module_local())
        .def_readonly("t", &PyOdeResult<Tt>::t)
        .def_readonly("y", &PyOdeResult<Tt>::y)
        .def_readonly("diverges", &PyOdeResult<Tt>::diverges)
        .def_readonly("is_stiff", &PyOdeResult<Tt>::is_stiff)
        .def_readonly("runtime", &PyOdeResult<Tt>::runtime);
    }

template<class Tt, class Ty>
void define_lowlevel_ode(py::module& m, const ode_f<Tt, Ty>& func_ptr){
    define_ode_module<Tt, Ty>(m);

    m.def("ode", [func_ptr]() {
        return PyOde<Tt, Ty>(func_ptr);
    });
}


#endif

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o _lowlevelode$(python3-config --extension-suffix)