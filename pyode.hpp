#ifndef PYODE_HPP
#define PYODE_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <ode.hpp>

namespace py = pybind11;


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


template<class Tt, size_t N = 0, bool raw_ode = true, bool raw_event = true>
class PyOde : public ODE<Tt, N, raw_ode, raw_event>{

    public:
        PyOde(ode_t<Tt, N, raw_ode> df): ODE<Tt, N, raw_ode, raw_event>(df) {}

        PyOde<Tt, N, raw_ode, raw_event> copy() const;
        
        const PyOdeResult<Tt> pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const;
        
        py::list pysolve_all(const py::list&, int threads) const;

        static py::list py_dsolve_all(const py::list& data, int threads);

};


template<class Tt, size_t N, bool raw_ode, bool raw_event>
struct PyOdeSet{

    PyOde<Tt, N, raw_ode, raw_event> ode;
    PyOdeArgs<Tt> params;

};
#pragma GCC visibility pop

template<class ArrayType, class T>
ArrayType toCPP_Array(const py::array& A);

template<class T, size_t N>
py::array_t<T> to_numpy(const vec<T, N>& array, const std::vector<size_t>& shape);

template<class Tt, size_t N>
std::vector<Tt> flatten(const std::vector<vec<Tt, N>>&);

template<class Tt, size_t N>
OdeArgs<Tt, N, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams);

template<class Tt, size_t N>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, N>& res);


/*
------------------------------------------------------------------------------------
-------------------------- IMPLEMENTATIONS -----------------------------------------
------------------------------------------------------------------------------------
*/

template<class ArrayType, class T>
ArrayType toCPP_Array(const py::array& A){
    size_t n = A.size();
    ArrayType res(n);

    const T* data = static_cast<const T*>(A.data());

    for (size_t i=0; i<n; i++){
        res[i] = data[i].template cast<T>();
    }
    return res;
}

template<class T, size_t N>
py::array_t<T> to_numpy(const vec<T, N>& array, const std::vector<size_t>& shape){
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


template<class Tt, size_t N>
OdeArgs<Tt, N, false> to_OdeArgs(const PyOdeArgs<Tt>& pyparams){
    using Ty = vec<Tt, N>;
    ICS<Tt, N> ics = {pyparams.ics[0].template cast<Tt>(), toCPP_Array<Ty, Tt>(pyparams.ics[1])};
    Tt t = pyparams.x;
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
        getcond = [pyparams](const Tt& t1, const Ty& f1, const Tt& t2, const Ty& f2) -> bool {
            bool res = pyparams.getcond(t1, t2, to_numpy(f1, {f1.size()}), to_numpy(f2, {f2.size()})).equal(py::bool_(true));
            return res;
        };
    }
    
    if (!pyparams.breakcond.is(py::none())) {
        breakcond = [pyparams](const Tt& t1, const Ty& f1, const Tt& t2, const Ty& f2) -> bool {
            bool res = pyparams.breakcond(t1, t2, to_numpy(f1, {f1.size()}), to_numpy(f2, {f2.size()})).equal(py::bool_(true));
            return res;
        };
    }

    OdeArgs<Tt, N, false> res = {ics, t, h, rtol, atol, cutoff_step, method, max_frames, args, getcond, breakcond};
    return res;
}


template<class Tt, size_t N>
PyOdeResult<Tt> to_PyOdeResult(const OdeResult<Tt, N>& res){

    std::vector<Tt> f_flat = flatten(res.y);
    size_t nd = res.y[0].size();
    size_t nt = res.y.size();

    PyOdeResult<Tt> pyres{res.x, f_flat, to_numpy(res.t, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.is_stiff, res.runtime};

    return pyres;
}


/*
----------------------CLASS IMPLEMENTATION---------------------
*/

template<class Tt, size_t N, bool raw_ode, bool raw_event>
const PyOdeResult<Tt> PyOde<Tt, N, raw_ode, raw_event>::pysolve(const py::tuple& ics, const Tt& t, const Tt& dt, const Tt& rtol, const Tt& atol, const Tt& cutoff_step, const py::str& method, const size_t& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const {

    const PyOdeArgs<Tt> pyparams = {ics, t, dt, rtol, atol, cutoff_step, method, max_frames, pyargs, getcond, breakcond};
    OdeArgs<Tt, N, raw_event> ode_args = to_OdeArgs<Tt, N>(pyparams);

    OdeResult<Tt, N> res = ODE<Tt, N, raw_ode, raw_event>::solve(ode_args);

    PyOdeResult<Tt> pyres = to_PyOdeResult(res);

    return pyres;
}

#endif