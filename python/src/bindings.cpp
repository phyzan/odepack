#include "../../include/pyode/PYODE.hpp"

namespace ode{

PYBIND11_MODULE(odesolvers, m) {

    py::class_<PyFuncWrapper>(m, "LowLevelFunction")
        .def(py::init<py::capsule, py::ssize_t, py::array_t<py::ssize_t>, py::ssize_t, py::str>(),
            py::arg("pointer"),
            py::arg("input_size"),
            py::arg("output_shape"),
            py::arg("Nargs"),
            py::arg("scalar_type")="double")
        .def("__call__", &PyFuncWrapper::call, py::arg("t"), py::arg("q"))
        .def_property_readonly("scalar_type", [](const PyFuncWrapper& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<EventOptions>(m, "EventOpt")
        .def(py::init<const std::string&, int, bool, int>(),
             py::arg("name"),
             py::arg("max_events") = -1,
             py::arg("terminate") = false,
             py::arg("period") = 1)
        .def_readwrite("name", &EventOptions::name)
        .def_readwrite("max_events", &EventOptions::max_events)
        .def_readwrite("terminate", &EventOptions::terminate)
        .def_readwrite("period", &EventOptions::period);


    py::class_<PyEvent>(m, "Event")
        .def_property_readonly("name", &PyEvent::name)
        .def_property_readonly("hides_mask", &PyEvent::hide_mask)
        .def_property_readonly("scalar_type", [](const PyEvent& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyPrecEvent, PyEvent>(m, "PreciseEvent")
        .def(py::init<std::string, py::object, int, py::object, bool, py::object, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("when"),
            py::arg("direction")=0,
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("event_tol")=1e-12,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("event_tol", &PyPrecEvent::event_tol);

    py::class_<PyPerEvent, PyEvent>(m, "PeriodicEvent")
        .def(py::init<std::string, py::object, py::object, bool, std::string, size_t, size_t>(),
            py::arg("name"),
            py::arg("period"),
            py::arg("mask")=py::none(),
            py::arg("hide_mask")=false,
            py::arg("scalar_type") = "double",
            py::arg("__Nsys")=0,
            py::arg("__Nargs")=0)
        .def_property_readonly("period", &PyPerEvent::period);


    py::class_<PySolver>(m, "OdeSolver")
        .def_property_readonly("t", &PySolver::t)
        .def_property_readonly("q", &PySolver::q)
        .def_property_readonly("t_old", &PySolver::t_old)
        .def_property_readonly("q_old", &PySolver::q_old)
        .def_property_readonly("stepsize", &PySolver::stepsize)
        .def_property_readonly("diverges", &PySolver::diverges)
        .def_property_readonly("is_dead", &PySolver::is_dead)
        .def_property_readonly("Nsys", &PySolver::Nsys)
        .def_property_readonly("n_evals_rhs", &PySolver::n_evals_rhs)
        .def_property_readonly("status", &PySolver::message)
        .def_property_readonly("at_event", &PySolver::py_at_event)
        .def("event_located", &PySolver::py_event_located, py::arg("event"))
        .def("show_state", &PySolver::show_state,
            py::arg("digits") = 8
        )
        .def("advance", &PySolver::advance)
        .def("advance_to_event", &PySolver::advance_to_event)
        .def("advance_until", &PySolver::advance_until, py::arg("t"))
        .def("reset", &PySolver::reset)
        .def("set_ics", &PySolver::set_ics, py::arg("t0"), py::arg("q0"), py::arg("stepsize")=0, py::arg("direction")=0)
        .def("resume", &PySolver::resume)
        .def("copy", &PySolver::copy)
        .def_property_readonly("scalar_type", [](const PySolver& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyRK23, PySolver>(m, "RK23")
        .def(py::init<PyRK23>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyRK45, PySolver>(m, "RK45")
        .def(py::init<PyRK45>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyDOP853, PySolver>(m, "DOP853")
        .def(py::init<PyDOP853>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyBDF, PySolver>(m, "BDF")
        .def(py::init<PyBDF>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");


    py::class_<PyRK4, PySolver>(m, "RK4")
        .def(py::init<PyRK4>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("scalar_type")="double");

    py::class_<PyVarSolver, PySolver>(m, "VariationalSolver")
        .def(py::init<PyVarSolver>(), py::arg("solver"))
        .def(py::init<py::object, py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, std::string, std::string>(),
            py::arg("f"),
            py::arg("jac"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def_property_readonly("logksi", &PyVarSolver::py_logksi)
        .def_property_readonly("lyap", &PyVarSolver::py_lyap)
        .def_property_readonly("t_lyap", &PyVarSolver::py_t_lyap)
        .def_property_readonly("delta_s", &PyVarSolver::py_delta_s);

    py::class_<PyOdeResult>(m, "OdeResult")
        .def(py::init<PyOdeResult>(), py::arg("result"))
        .def_property_readonly("t", &PyOdeResult::t)
        .def_property_readonly("q", &PyOdeResult::q)
        .def_property_readonly("event_map", &PyOdeResult::event_map)
        .def("event_data", &PyOdeResult::event_data, py::arg("event"))
        .def_property_readonly("diverges", &PyOdeResult::diverges)
        .def_property_readonly("success", &PyOdeResult::success)
        .def_property_readonly("runtime", &PyOdeResult::runtime)
        .def_property_readonly("message", &PyOdeResult::message)
        .def("examine", &PyOdeResult::examine)
        .def_property_readonly("scalar_type", [](const PyOdeResult& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyOdeSolution, PyOdeResult>(m, "OdeSolution")
        .def(py::init<PyOdeSolution>(), py::arg("result"))
        .def("__call__", [](const PyOdeSolution& self, const py::object& t){return self(t);});

    py::class_<PyODE>(m, "LowLevelODE")
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyODE>(), py::arg("ode"))
        .def("rhs", &PyODE::call_Rhs, py::arg("t"), py::arg("q"))
        .def("jac", &PyODE::call_Jac, py::arg("t"), py::arg("q"))
        .def("solver", &PyODE::solver_copy, py::keep_alive<0, 1>())
        .def("integrate", &PyODE::py_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("integrate_until", &PyODE::py_integrate_until,
            py::arg("t"),
            py::kw_only(),
            py::arg("t_eval")=py::none(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("rich_integrate", &PyODE::py_rich_integrate,
            py::arg("interval"),
            py::kw_only(),
            py::arg("event_options")=py::tuple(),
            py::arg("max_prints")=0)
        .def("copy", &PyODE::copy)
        .def("reset", &PyODE::reset)
        .def("clear", &PyODE::clear)
        .def("event_data", &PyODE::event_data, py::arg("event"))
        .def_property_readonly("t", &PyODE::t_array)
        .def_property_readonly("q", &PyODE::q_array)
        .def_property_readonly("event_map", &PyODE::event_map)
        .def_property_readonly("Nsys", &PyODE::Nsys)
        .def_property_readonly("runtime", &PyODE::runtime)
        .def_property_readonly("diverges", &PyODE::diverges)
        .def_property_readonly("is_dead", &PyODE::is_dead)
        .def_property_readonly("scalar_type", [](const PyODE& self){return SCALAR_TYPE[self.scalar_type];});

    py::class_<PyVarODE, PyODE>(m, "VariationalLowLevelODE")
        .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object, py::object, py::object, py::object, py::object, int, py::iterable, py::iterable, py::str, std::string>(),
            py::arg("f"),
            py::arg("t0"),
            py::arg("q0"),
            py::arg("period"),
            py::kw_only(),
            py::arg("jac")=py::none(),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("args")=py::tuple(),
            py::arg("events")=py::tuple(),
            py::arg("method")="RK45",
            py::arg("scalar_type")="double")
        .def(py::init<PyVarODE>(), py::arg("ode"))
        .def_property_readonly("t_lyap", &PyVarODE::py_t_lyap)
        .def_property_readonly("lyap", &PyVarODE::py_lyap)
        .def_property_readonly("kicks", &PyVarODE::py_kicks)
        .def("copy", &PyVarODE::copy);

    py::class_<PyVecFieldBase>(m, "SampledVectorField")
        .def("streamline", &PyVecFieldBase::py_streamline,
            py::arg("q0"),
            py::arg("length"),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("t_eval")=py::none(),
            py::arg("method")="RK45"
        )
        .def("get_ode", &PyVecFieldBase::py_streamline_ode,
            py::arg("q0"),
            py::arg("rtol")=1e-12,
            py::arg("atol")=1e-12,
            py::arg("min_step")=0.,
            py::arg("max_step")=py::none(),
            py::arg("stepsize")=0.,
            py::arg("direction")=1,
            py::arg("method")="RK45",
            py::arg("normalized")=false, py::keep_alive<0, 1>()
        )
        .def("streamplot_data", &PyVecFieldBase::py_streamplot_data,
            py::arg("max_length"),
            py::arg("ds"),
            py::arg("density")=30
        );

    py::class_<PyVecField2D, PyVecFieldBase>(m, "SampledVectorField2D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("vx"),
            py::arg("vy"))
        .def_property_readonly("x", &PyVecField2D::py_x<0>)
        .def_property_readonly("y", &PyVecField2D::py_x<1>)
        .def_property_readonly("vx", &PyVecField2D::py_vx<0>)
        .def_property_readonly("vy", &PyVecField2D::py_vx<1>)
        .def("get_vx", &PyVecField2D::py_vx_at<0, double, double>, py::arg("x"), py::arg("y"))
        .def("get_vy", &PyVecField2D::py_vx_at<1, double, double>, py::arg("x"), py::arg("y"))
        .def("__call__", &PyVecField2D::py_vector<double, double>, py::arg("x"), py::arg("y"))
        .def("in_bounds", &PyVecField2D::py_in_bounds<double, double>, py::arg("x"), py::arg("y"));

    py::class_<PyVecField3D, PyVecFieldBase>(m, "SampledVectorField3D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("vx"),
            py::arg("vy"),
            py::arg("vz"))
        .def_property_readonly("x", &PyVecField3D::py_x<0>)
        .def_property_readonly("y", &PyVecField3D::py_x<1>)
        .def_property_readonly("z", &PyVecField3D::py_x<2>)
        .def_property_readonly("vx", &PyVecField3D::py_vx<0>)
        .def_property_readonly("vy", &PyVecField3D::py_vx<1>)
        .def_property_readonly("vz", &PyVecField3D::py_vx<2>)
        .def("get_vx", &PyVecField3D::py_vx_at<0, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("get_vy", &PyVecField3D::py_vx_at<1, double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("__call__", &PyVecField3D::py_vector<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("in_bounds", &PyVecField3D::py_in_bounds<double, double, double>, py::arg("x"), py::arg("y"), py::arg("z"));

    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

    m.def("advance_all", &py_advance_all, py::arg("solvers"), py::arg("t_goal"), py::arg("threads")=-1, py::arg("display_progress")=false);

    py::class_<PyScalarField<1>>(m, "SampledScalarField1D")
        .def(py::init<py::array_t<double>, py::array_t<double>>(),
            py::arg("f"),
            py::arg("x"));

    py::class_<PyScalarField<2>>(m, "SampledScalarField2D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("f"),
            py::arg("x"),
            py::arg("y"));

    py::class_<PyScalarField<3>>(m, "SampledScalarField3D")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>(),
            py::arg("f"),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"));

    py::class_<PyScatteredField>(m, "ScatteredScalarField")
        .def(py::init<py::array_t<double>, py::array_t<double>>(),
            py::arg("points"),
            py::arg("values"))
        .def_property_readonly("points", &PyScatteredField::py_points)
        .def_property_readonly("values", &PyScatteredField::py_values)
        .def("__call__", &PyScatteredField::py_value_at);

#ifdef MPREAL
    m.def("set_mpreal_prec",
      &mpfr::mpreal::set_default_prec,
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", &mpfr::mpreal::get_default_prec);
#else
    m.def("set_mpreal_prec",
      [](size_t){
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
        throw py::value_error("Current installation does not support mpreal for arbitrary precision");
      });
#endif
}

} // namespace ode
