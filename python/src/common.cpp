#include "include/pyode/pyode.hpp"


void py_integrate_all(const py::object& list, double interval, const py::object& t_eval, const py::iterable& event_options, int threads, bool display_progress){
    // Separate lists for each numeric type
    std::vector<ODE<double, 0>*> array_double;
    std::vector<ODE<float, 0>*> array_float;
    std::vector<ODE<long double, 0>*> array_longdouble;
    std::vector<ODE<mpfr::mpreal, 0>*> array_mpreal;

    // Iterate through the list and identify each PyODE type
    for (const py::handle& item : list) {
        // Try double
        try {
            auto& pyode = item.cast<PyODE<double>&>();
            array_double.push_back(pyode.ode);
            continue;
        } catch (const py::cast_error&) {}

        // Try float
        try {
            auto& pyode = item.cast<PyODE<float>&>();
            array_float.push_back(pyode.ode);
            continue;
        } catch (const py::cast_error&) {}

        // Try long double
        try {
            auto& pyode = item.cast<PyODE<long double>&>();
            array_longdouble.push_back(pyode.ode);
            continue;
        } catch (const py::cast_error&) {}

        // Try mpreal
        try {
            auto& pyode = item.cast<PyODE<mpfr::mpreal>&>();
            array_mpreal.push_back(pyode.ode);
            continue;
        } catch (const py::cast_error&) {}

        // If none of the types matched, throw an error
        throw py::value_error("List item is not a recognized PyODE object type (double, float, long double, or mpreal).");
    }

    // Convert event_options once (it's not templated)
    auto options = to_Options(event_options);

    // Call integrate_all for each type group that has elements
    if (!array_double.empty()) {
        integrate_all<double, 0>(array_double, interval, to_step_sequence<double>(t_eval), options, threads, display_progress);
    }
    if (!array_float.empty()) {
        integrate_all<float, 0>(array_float, static_cast<float>(interval), to_step_sequence<float>(t_eval), options, threads, display_progress);
    }
    if (!array_longdouble.empty()) {
        integrate_all<long double, 0>(array_longdouble, static_cast<long double>(interval), to_step_sequence<long double>(t_eval), options, threads, display_progress);
    }

    if (!array_mpreal.empty()) {
        integrate_all<mpfr::mpreal, 0>(array_mpreal, mpfr::mpreal(interval), to_step_sequence<mpfr::mpreal>(t_eval), options, threads, display_progress);
    }
}

PYBIND11_MODULE(common, m) {
    define_event_opt(m);
    
    m.def("integrate_all", &py_integrate_all, py::arg("ode_array"), py::arg("interval"), py::arg("t_eval")=py::none(), py::arg("event_options")=py::tuple(), py::arg("threads")=-1, py::arg("display_progress")=false);

    m.def("set_mpreal_prec",
      [](int bits) {
          mpfr::mpreal::set_default_prec(bits);
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
          return mpfr::mpreal::get_default_prec();
      });
}