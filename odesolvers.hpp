#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include <eigen3/Eigen/Dense>
#include "tools.hpp"

template<class Tt, size_t N = 0, bool raw = true>
struct SolverArgs{

    ode_t<Tt, N, raw> f;
    ICS<Tt, N> ics;
    Tt t;
    Tt h;
    Tt rtol;
    Tt atol;
    Tt h_min;
    std::vector<Tt> args;
};

template<class Tt, size_t N = 0, bool raw = true>
SolverArgs<Tt, N, raw> to_SolverArgs(const ode_t<Tt, N, raw>& f, const OdeArgs<Tt, N>& args){
    return {f, args.ics, args.t, args.h, args.rtol, args.atol, args.cutoff_step, args.args};
}



template<class Tt, size_t N = 0, bool raw = true>
class OdeSolver{


public:

    using Ty = vec<Tt, N>;
    using Callable = ode_t<Tt, N, raw>;

    static constexpr Tt MAX_FACTOR = Tt(10);
    static constexpr Tt SAFETY = Tt(9)/10;
    static constexpr Tt MIN_FACTOR = Tt(2)/10;

    const Callable f;
    const Tt t_max;
    const Tt min_h;
    const std::vector<Tt> args;
    const int direction;
    const size_t n;
    const Tt rtol;
    const Tt atol;

private:

    Tt _h;
    Tt _t;
    Ty _y;
    bool _is_running = true;
    bool _diverges = false;
    bool _is_stiff = false;
    size_t neval=0;

public:

    //ACCESSORS
    const Tt& t_now() const {return _t;}

    const Ty& y_now() const {return _y;}

    const Tt& h_now() const {return _h;}

    const bool& is_running() const {return _is_running;}

    SolverState<Tt, N> state() const {
        SolverState<Tt, N> res = {_t, _y, _diverges, _is_stiff, _is_running, neval};
        return res;
    }

    //MODIFIERS
    void stop() {_is_running = false;}

    bool advance_by(const Tt& h);

    //step(...) must NOT depend on current state
    virtual Ty step(const Tt& t_old, const Ty& y_old, const Tt& h) const = 0;

    virtual bool advance() = 0;

    void set_ics(const Tt& t0, const Ty& y0){
        if (_is_running){
            _t = t0;
            _y = y0;
        }
        else{
            throw std::runtime_error("Cannot set new ics to solver, as it has already finished integrating.");
        }
    }
    ~OdeSolver() = default;

protected:

    OdeSolver(const SolverArgs<Tt, N, raw>& S): f(S.f), t_max(S.t), min_h(S.h_min), args(S.args), direction( S.h > 0 ? 1 : -1), n(S.ics.y0.size()), _h(S.h), _t(S.ics.t0), _y(S.ics.y0), rtol(S.rtol), atol(S.atol) {}

    bool _update(const Tt& t_new, const Ty& y_new, const Tt& h_next);

private:

    OdeSolver operator=(const OdeSolver&) = delete;
    OdeSolver(const OdeSolver& other) = default;
};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/


template<class Tt, size_t N, bool raw>
bool OdeSolver<Tt, N, raw>::advance_by(const Tt& h){
    return _update(_t+h, step(_t, _y, h), h);
}


template<class Tt, size_t N, bool raw>
bool OdeSolver<Tt, N, raw>::_update(const Tt& t_new, const OdeSolver<Tt, N, raw>::Ty& y_new, const Tt& h_next){

    bool success = true;
    if (! _is_running){
        success = false;
        throw std::runtime_error("Solver has finished integrating.");
    }

    bool _stop = false;
    Tt stepsize = h_next*direction;
    if (stepsize < 0){
        success = false;
        throw std::runtime_error("Wrong direction of integration");
    }

    if (stepsize <= min_h){
        _is_stiff = true;
        _stop = true;
        success = false;
    }
    if (!y_new.isFinite().all()){
        _diverges = true;
        _stop = true;
        success = false;
    }
    else if (t_new*direction >= t_max*direction){
        _y = this->step(_t, _y, t_max-_t);
        _t = t_max;
        _h = h_next;
        neval++;
        _stop = true;
    }
    else{
        _t = t_new;
        _y = y_new;
        _h = h_next;
        neval++;
    }

    if (_stop) stop();

    return success;
}

#endif