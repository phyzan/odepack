#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <array>
#include <string>
#include <eigen3/Eigen/Dense>
#include "tools.hpp"



template<class ODS, class Tt, size_t N = 0>
class OdeSolver{


public:

    using Ty = std::conditional_t<(N == 0),
    Eigen::Array<Tt, Eigen::Dynamic, 1>,
    Eigen::Array<Tt, N, 1>>;

    static constexpr Tt MAX_FACTOR = Tt(10);
    static constexpr Tt SAFETY = Tt(9)/10;
    static constexpr Tt MIN_FACTOR = Tt(2)/10;

public:

    const ode<Tt, Ty> f;
    const Tt t_max;
    const Tt min_h;
    const std::vector<Tt> args;
    const int direction;
    const size_t n;

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

    SolverState<Tt, Ty> state() const {
        SolverState<Tt, Ty> res = {_t, _y, _diverges, _is_stiff, _is_running, neval};
        return res;
    }

    //MODIFIERS
    void stop() {_is_running = false;}

    void advance_by(const Tt& h);

    Ty step(const Tt& t_old, const Ty& y_old, const Tt& h) {return static_cast<ODS*>(this)->step(t_old, y_old, h);}

    void advance(){static_cast<ODS*>(this)->advance();}

protected:

    OdeSolver(ode<Tt, Ty> func, const Ty& y0, const Tt (&span)[2], const Tt& h, const Tt& min_h, const std::vector<Tt>& args): f(func), t_max(span[1]), min_h(min_h), args(args), direction( h > 0 ? 1 : -1), n(y0.size()), _h(h), _t(span[0]), _y(y0) {}//make constructor in case F is not lambda or std::function, to reduce overhead. F might need to be a template parameter
    

    ~OdeSolver() = default;

    void _update(const Tt& t_new, const Ty& y_new, const Tt& h_next);

private:

    OdeSolver(const OdeSolver& other) = default;
    OdeSolver<Tt, Ty> operator=(const OdeSolver<Tt, Ty>&) = delete;
};


/*
------------------------------------------------------------------------------
-----------------------------IMPLEMENTATIONS----------------------------------
------------------------------------------------------------------------------
*/


template<class ODS, class Tt, size_t N>
void OdeSolver<ODS, Tt, N>::advance_by(const Tt& h){
    _update(_t+h, step(_t, _y, h));
}


template<class ODS, class Tt, size_t N>
void OdeSolver<ODS, Tt, N>::_update(const Tt& t_new, const OdeSolver<ODS, Tt, N>::Ty& y_new, const Tt& h_next){
    if (! _is_running){
        throw std::runtime_error("Solver has finished integrating.");
    }

    bool _stop = false;
    Tt stepsize = h_next*direction;
    if (stepsize < 0){
        throw std::runtime_error("Wrong direction of integration");
    }

    if (stepsize <= min_h){
        _is_stiff = true;
        _stop = true;
    }
    if (!y_new.isFinite().all()){
        _diverges = true;
        _stop = true;
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
}

#endif