#ifndef STATES_HPP
#define STATES_HPP

#include "tools.hpp"

template<typename T, int N>
class State{

    //only provides access, no changes. Derived solver has setters

public:

    virtual State<T, N>* clone() const = 0;

    inline T h() const;

    inline const T& habs()const;

    inline const T& t()const{
        return _t;
    }

    inline const vec<T, N>& vector()const;

    inline virtual const vec<T, N>& exposed_vector() const{
        return _q;
    }

    inline const int& direction()const{
        return _direction;
    }

    inline vec<T, N> local_error() const{
        return _error.cwiseAbs(); //might need to be divided by sqrt(vector size)
    }

    virtual bool is_temporary() const{
        return false;
    }

    virtual ~State(){}


protected:

    State(const T& t, const vec<T, N>& q, const T& h):_t(t), _q(q), _habs(abs(h)), _direction(sgn(h)), _error(q.size()){
        _error.setZero();
    }

    State(){
        _q.setZero();
        _error.setZero();
    }

    DEFAULT_RULE_OF_FOUR(State);

    T _t = 0; //current time
    vec<T, N> _q; //current vector
    T _habs=0; //absolute stepsize to be used for the next step
    int _direction = 0;
    vec<T, N> _error;
};



template<typename T, int N, typename Derived, typename STATE, typename INTERPOLATOR>
class DerivedSolver;


template<typename T, int N, class Derived>
class DerivedState : public State<T, N>{

    //virtual methods can be replaces with stack polymorphism

    template<class, int, typename, typename, typename>
    friend class DerivedSolver;

public:

    State<T, N>* clone() const final{
        return new Derived(static_cast<const Derived&>(*this));
    }

    virtual void adjust(const T& h_abs, const T& dir, const vec<T, N>& diff);

    virtual bool resize_step(T& factor, const T& min_step=0, const T& max_step=inf<T>()){
        //TODO: convert runtime polymorphism to static, since it already uses crtp
        return _validate_step_resize(factor, min_step, max_step);
    }

protected:

    DerivedState(const T& t, const vec<T, N>& q, const T& h) : State<T, N>(t, q, h){}
    
    DEFAULT_RULE_OF_FOUR(DerivedState);
    
    bool _validate_step_resize(T& factor, const T& min_step, const T& max_step);
};



template<typename T, int N>
class ViewState final : public State<T, N>{

public:

    ViewState():State<T, N>(){
        _q_exposed.setZero();
    }

    ViewState(const T& t, const vec<T, N>& q, const T& h=0) : State<T, N>(t, q, abs(h)), _q_exposed(q) {
        _q_exposed_ptr = &(this->_q);
        this->_direction = (h > 0) ? 1 : ( (h<0) ? -1 : 0);
    }

    ViewState(const T& t, const vec<T, N>& q_true, const vec<T, N>& q_exposed, const T& h=0) : State<T, N>(t, q_true, h), _q_exposed(q_exposed) {
        _q_exposed_ptr = &q_exposed;
    }

    ViewState(const ViewState<T, N>& other) : State<T, N>(other), _q_exposed(other._q_exposed){
        _set_ptr(other);
    }

    
    ViewState(ViewState&& other) : State<T, N>(std::move(other)), _q_exposed(std::move(other._q_exposed)){
        _set_ptr(other);
        other._q_exposed_ptr = nullptr;
    }

    ViewState<T, N>& operator=(const ViewState<T, N>& other){
        if (&other != this){
            State<T, N>::operator=(other);
            _q_exposed = other._q_exposed;
            _set_ptr(other);
        }
        return *this;
    }

    ViewState<T, N>& operator=(ViewState<T, N>&& other){
        if (&other != this){
            State<T, N>::operator=(std::move(other));
            _q_exposed = std::move(other._q_exposed);
            _set_ptr(other);
            other._q_exposed_ptr = nullptr;
        }
        return *this;
    }

    const vec<T, N>& exposed_vector() const final{
        return *_q_exposed_ptr;
    }

    ViewState<T, N>* clone() const final{
        return new ViewState<T, N>(*this);
    }

    bool is_temporary() const final{
        return true;
    }

    void set(const T& t, const vec<T, N>& q){
        this->_t = t;
        this->_q = q;
        _q_exposed_ptr = &(this->_q);
        this->_error.resize(q.size());
        this->_error.setZero();
    }

    void set(const T& t, const vec<T, N>& q_true, const vec<T, N>& q_exposed){
        this->_t = t;
        this->_q = q_true;
        this->_q_exposed = q_exposed;
        this->_error.resize(q_true.size());
        this->_error.setZero();
        _q_exposed_ptr = &_q_exposed;
    }

private:

    void _set_ptr(const ViewState<T, N>& other){
        if (other._q_exposed_ptr == &other._q_exposed){
            _q_exposed_ptr = &_q_exposed;
        }
        else{
            _q_exposed_ptr = &(this->_q);
        }
    }

    vec<T, N> _q_exposed;
    const vec<T, N>* _q_exposed_ptr = nullptr;

};

























template<typename T, int N, class Derived>
inline void DerivedState<T, N, Derived>::adjust(const T& h_abs, const T& dir, const vec<T, N>&){
    this->_habs = h_abs;
    this->_direction = sgn(dir);
}


template<typename T, int N>
inline T State<T, N>::h()const{
    return this->_habs*this->_direction;
}


template<typename T, int N>
inline const T& State<T, N>::habs()const{
    return this->_habs;
}


template<typename T, int N>
inline const vec<T, N>& State<T, N>::vector()const{
    return this->_q;
}


template<typename T, int N, class Derived>
bool DerivedState<T, N, Derived>::_validate_step_resize(T& factor, const T& min_step, const T& max_step){
    bool res = false;
    if (this->_habs*factor < min_step){
        factor = min_step/this->_habs;
        this->_habs = min_step;
    }
    else if (this->_habs*factor > max_step){
        factor = max_step/this->_habs;
        this->_habs = max_step;
    }
    else{
        this->_habs *= factor;
        res = true;
    }
    return res;
}


#endif