#ifndef ODE_HPP
#define ODE_HPP

#include <variant>
#include "adaptive_rk.hpp"

template<class Tt, size_t N, bool raw>
using OdsVariant = std::variant<RK23<Tt, N, raw>, RK45<Tt, N, raw>>;

template<class Tt, size_t N = 0, bool raw_ode = true, bool raw_event = true>
class ODE{

public:

    ode_t<Tt, N, raw_ode> f;

    ODE(ode_t<Tt, N, raw_ode> ode = nullptr) : f(ode) {};

    const OdeResult<Tt, N> solve(const OdeArgs<Tt, N>& args) const;

    const std::vector<OdeResult<Tt, N>> solve_all(const std::vector<OdeArgs<Tt, N>>& args, int threads=-1) const;
};

template<class Tt, size_t N, bool raw=true>
struct OdeSet{

    ODE<Tt, N, raw> ode;
    OdeArgs<Tt, N> params;

};

template<class Tt, size_t N, bool raw=true>
std::vector<OdeResult<Tt, N>> dsolve_all(const std::vector<OdeSet<Tt, N, raw>>& data, int threads);


/*
-----------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS-------------------------------
-----------------------------------------------------------------------
*/


template<class Tt, size_t N, bool raw_ode, bool raw_event>
const OdeResult<Tt, N> ODE<Tt, N, raw_ode, raw_event>::solve(const OdeArgs<Tt, N>& args) const{


    RK23<Tt> solver(f, args.y0, args.t_span, args.h, args.min_h, args.args, args.abs_tol, args.rel_tol);
    while (solver.is_running()){
        solver.advance();
    }
    return solver.state();
}






#endif