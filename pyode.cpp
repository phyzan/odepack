#include <src/pyode.hpp>

template<class T>
using StackVec = vec<T, 4>;

StackVec<double> ODE_FUNC(const double& t, const StackVec<double>& q, const std::vector<double>& args){
        return {imag(exp((complex<double>(0, 1) - complex<double>(0, 1))*cos(args[0]*t)*sin(args[0]*t))/q[0]), imag(exp((complex<double>(0, 1) - complex<double>(0, 1))*cos(args[0]*t)*sin(args[0]*t))/q[1]), imag(-exp((complex<double>(0, 1) - complex<double>(0, 1))*cos(args[0]*t)*sin(args[0]*t))/pow(q[0], 2))*q[2], imag(-exp((complex<double>(0, 1) - complex<double>(0, 1))*cos(args[0]*t)*sin(args[0]*t))/pow(q[1], 2))*q[3]};
}

PyODE<double, StackVec<double>>* GetOde(const double& t0, py::array q0, const double& stepsize, const double& rtol, const double& atol, const double& min_step, py::tuple args, py::str method, const double& event_tol, py::str savedir){
        return new PyODE<double, StackVec<double>>(ODE_FUNC, t0, toCPP_Array<double, StackVec<double>>(q0), stepsize, rtol, atol, min_step, toCPP_Array<double, std::vector<double>>(args), method.cast<std::string>(), event_tol, {Event<double, StackVec<double>>("checkpoint", nullptr, nullptr, 0.5, -1e-09, nullptr), Event<double, StackVec<double>>("transformation", nullptr, nullptr, 0.5, 0, [](const double& t, const StackVec<double>& q, const std::vector<double>& args) -> StackVec<double> {return {q[0], q[1], q[2]*pow(pow(q[2], 2) + pow(q[3], 2), -1./2.), q[3]*pow(pow(q[2], 2) + pow(q[3], 2), -1./2.)};})}, {}, savedir.cast<std::string>());
}

PYBIND11_MODULE(ODE_MODULE_0, m){
        define_ode_module<double, vec<double, 4>>(m);
        m.def("get_ode", GetOde);
}