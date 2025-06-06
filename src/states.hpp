#ifndef STATES_HPP
#define STATES_HPP

#include "tools.hpp"

template<class T, int N>
class State{

public:

    virtual State<T, N>* clone() const = 0;

    inline void set_direction(const T& dir);
    
    virtual void adjust(const T& h_abs, const T& dir, const vec<T, N>& diff);

    virtual bool resize_step(T& factor, const T& min_step=0, const T& max_step=inf<T>()){
        return _validate_step_resize(factor, min_step, max_step);
    }

    inline T h() const;

    inline const T& habs()const;

    inline const T& t()const{
        return _t;
    }

    inline const vec<T, N>& vector()const;

    inline const int& direction()const{
        return _direction;
    }

    inline vec<T, N> local_error() const{
        return _error.cwiseAbs(); //might need to be divided by sqrt(vector size)
    }

    virtual void assign(const T& t, const vec<T, N>& q, const T& habs, const T& dir, const vec<T, N>& diff){
        //ideally should also assign a new error
        _t = t;
        _q = q;
        _habs = habs;
        set_direction(dir);
    }

    virtual ~State(){}


protected:

    State(const T& t, const vec<T, N>& q, const T& habs):_t(t), _q(q), _habs(habs), _error(q.size()){
        _error.setZero();
    }

    State<T, N>& operator=(const State<T, N>& other) = default;

    bool _validate_step_resize(T& factor, const T& min_step, const T& max_step);

    T _t; //current time
    vec<T, N> _q; //current vector
    T _habs; //absolute stepsize to be used for the next step
    int _direction = 0;
    vec<T, N> _error;
};



template<class T, int N, class STATE>
class DerivedSolver;


template<class T, int N, class Derived>
class DerivedState : public State<T, N>{

    friend DerivedSolver<T, N, Derived>;

public:

    State<T, N>* clone() const final{
        return new Derived(static_cast<const Derived&>(*this));
    }

protected:

    DerivedState(const T& t, const vec<T, N>& q, const T& habs) : State<T, N>(t, q, habs){}
    
};












template<class T, int N>
inline void State<T, N>::set_direction(const T& dir){
    this->_direction = (dir == 0) ? 0 : ( (dir > 0) ? 1 : -1);
}


template<class T, int N>
inline void State<T, N>::adjust(const T& h_abs, const T& dir, const vec<T, N>& diff){
    this->_habs = h_abs;
    set_direction(dir);
}


template<class T, int N>
inline T State<T, N>::h()const{
    return this->_habs*this->_direction;
}


template<class T, int N>
inline const T& State<T, N>::habs()const{
    return this->_habs;
}


template<class T, int N>
inline const vec<T, N>& State<T, N>::vector()const{
    return this->_q;
}


template<class T, int N>
bool State<T, N>::_validate_step_resize(T& factor, const T& min_step, const T& max_step){
    bool res = false;
    if (_habs*factor < min_step){
        factor = min_step/_habs;
        _habs = min_step;
    }
    else if (_habs*factor > max_step){
        factor = max_step/_habs;
        _habs = max_step;
    }
    else{
        _habs *= factor;
        res = true;
    }
    return res;
}



#endif