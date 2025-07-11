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

    State() = default;

    DEFAULT_RULE_OF_FOUR(State);

    T _t; //current time
    vec<T, N> _q; //current vector
    T _habs; //absolute stepsize to be used for the next step
    int _direction = 0;
    vec<T, N> _error;
};



template<typename T, int N, class Derived, class STATE>
class DerivedSolver;


template<typename T, int N, class Derived>
class DerivedState : public State<T, N>{

    //virtual methods can be replaces with stack polymorphism

    template<class, int, class, class>
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

    ViewState() = default;

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




















template<typename T>
class Interval{

public:

    Interval(const T& a, const T& b, const int& left_bdr=0, const int& right_bdr=0) : _a(a), _b(b), _h(b-a), _left(sgn(left_bdr)), _right(sgn(right_bdr)), _dir(sgn(b-a)){
        if (a == b){
            throw std::runtime_error("Zero interval not allowed");
        }
    }

    Interval(const T& x) : _a(x), _b(x), _left(1), _right(1), _is_point(true) {}

    DEFAULT_RULE_OF_FOUR(Interval);

    inline bool contains(const T& t) const{
        return (t==_a) ? _left != -1 : ( (t==_b) ? _right != -1 : (_a*_dir < t*_dir) && (t*_dir < _b*_dir));
    }

    inline const T& start() const{
        return _a;
    }

    inline const T& end() const{
        return _b;
    }

    inline const T& h() const{
        return _h;
    }

    inline const int& dir() const{
        return _dir;
    }

    inline const int& start_bdr() const{
        return _left;
    }

    inline const int& end_bdr() const{
        return _right;
    }

    inline const bool& is_point() const{
        return _is_point;
    }

    void open_end(){
        if (_is_point){
            throw std::runtime_error("Cannot open end boundary on a point interval");
        }
        else{
            _right = -1;
        }
    }

    inline void close_end(){
        _right = 1;
    }

    inline void close_start(){
        _left = 1;
    }

    void resize(const T& new_end){
        if (sgn(new_end-_a) == _dir){
            _b = new_end;
            _h = _b - _a;
        }
        else{
            throw std::runtime_error("Cannot resize interval");
        }
    }

    inline bool can_link_with(const Interval<T>& other) const{
        //boundaries and directions must match.
        return (_b == other._a) && (_right*other._left != 1) && !(_right == 0 && other._left == 0) && (_dir+other._dir != 0);
    }

    void link_with(Interval<T>& other) {
        if (!can_link_with(other)){
            throw std::runtime_error("Intervals cannot link");
        }

        if (_right == 0){
            _right = -other._left;
        }
        else if (other._left == 0){
            other._left = -_right;
        }
    }

    void match_bdrs(const Interval<T>& other){
        _left = other._left;
        _right = other._right;
    }

    std::string signature(const int& prec = 3) const{
        if (is_point()){
            return "[" + to_string(_a, prec) + "]";
        }
        else{
            std::string left = (_left == 1) ? "[" : (_left == -1 ? "(" : "|");
            std::string right = (_right == 1) ? "]" : (_right == -1 ? ")" : "|");
            return left + to_string(_a, prec) + ", " + to_string(_b, prec) + right;
        }
    }

private:

    T _a;
    T _b;
    T _h = 0;
    int _left = 0;
    int _right = 0;
    bool _is_point = false;
    int _dir = 0;
};


template<typename T, int N>
class LocalInterpolator final{
    //The purpose of this class is to provide a continuous output between the naturally adapted states
    //of an OdeSolver.
    
public:

    LocalInterpolator() = delete;

    LocalInterpolator(const T& t, const vec<T, N>& q);

    LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state);

    LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr);

    LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr, const std::array<T, 2>& range);

    DEFAULT_RULE_OF_FOUR(LocalInterpolator);

    int order() const;

    inline const Interval<T>& interval() const;

    void call(vec<T, N>& result, const T& t) const;

    void adjust(const T& t_end){
        const int& d = _max_interval.dir();
        if (_max_interval.start()*d <= t_end*d && t_end*d <= _max_interval.end()*d){
            _interval.resize(t_end);
        }
        else{
            throw std::runtime_error("Cannot adjust interval");
        }
    }

    inline bool can_link_with(const LocalInterpolator<T, N>& other) const{
        return _interval.can_link_with(other._interval);
    }

    inline void link_with(LocalInterpolator<T, N>& other){
        if (_interval.end() != other._interval.start()){
            throw std::runtime_error("Interpolators do not meet at the same point");
        }

        _interval.link_with(other._interval);
        _max_interval.match_bdrs(_interval);
    }

    inline void close_end(){
        _interval.close_end();
        _max_interval.close_end();
    }

    inline void close_start(){
        _interval.close_start();
        _max_interval.close_start();
    }

private:

    inline bool is_out_of_bounds(const T& t) const;

    Eigen::Matrix<T, N, -1> _coef_mat; //number of columns is equal to the interpolation order
    vec<T, N> _q_old;
    mutable vec<T> _theta;//column vector with "interpolation order" number of rows
    Interval<T> _interval;
    Interval<T> _max_interval;

};



template<typename T, int N>
class LinkedInterpolator final{

    template<class, int, class, class>
    friend class DerivedSolver;

public:

    LinkedInterpolator() = delete;

    LinkedInterpolator(const T& t, const vec<T, N>& q) : _interpolants{{t, q}}, _q(q.size()){}

    DEFAULT_RULE_OF_FOUR(LinkedInterpolator);

    inline void adjust_to(const T& t_new){
        _interpolants.back().adjust(t_new);
    }

    inline bool can_link_with(const LocalInterpolator<T, N>& interpolant) const {
        return _interpolants.back().can_link_with(interpolant) || _can_replace_last_with(interpolant);
    }

    void expand(const LocalInterpolator<T, N>& interpolant){

        if (_can_replace_last_with(interpolant)){
            _interpolants.back() = interpolant;
        }
        else if(_interpolants.back().can_link_with(interpolant)){
            _interpolants.push_back(interpolant);
        }
        else{
            throw std::runtime_error("Invalid interpolant");
        }

        if ((interpolant.interval().dir() != 0) && (_dir == 0)){
            _dir = interpolant.interval().dir();
        }
    }

    void call(vec<T, N>& result, const T& t) const{
        if (is_out_of_bounds(t)){
            throw std::runtime_error("Scalar " + to_string(t) + " is out of bounds in LinkedInterpolator with range ["+to_string(t_start())+", "+to_string(t_end())+"]");
        }
        else if (t == t_start()){
            _interpolants.front().call(result, t);
        }
        else if (t == t_end()){
            _interpolants.back().call(result, t);
        }
        else{
            size_t ind = _search_index(t);
            _interpolants[ind].call(result, t);
        }
    }

    vec<T, N> call(const T& t) const{
        this->call(_q, t);
        return _q;
    }

    inline bool is_out_of_bounds(const T& t) const {
        return (t*dir() < t_start()*dir()) || (t*dir() > t_end()*dir());
    }

    inline const T& t_start() const{
        return _interpolants[0].interval().start();
    }

    inline const T& t_end() const{
        return _interpolants.back().interval().end();
    }

    inline const int& dir() const{
        return _dir;
    }

    inline const std::vector<LocalInterpolator<T, N>>& interpolants() const{
        return _interpolants;
    }

    inline void open_end(){
        _interpolants.back().open_end();
    }

    inline void close_end(){
        _interpolants.back().close_end();
    }

    inline void close_start(){
        _interpolants.front().close_start();
    }

private:

    size_t _search_index(const T& t) const{
        //no need to check for out of bounds, it must have just been checked before this function call.
        //t must NOT be equal to the _t_end(). This case has already been taken care of.

        size_t left = 0;
        size_t right = _interpolants.size()-1;
        size_t mid;
        const int d = dir();

        while (right > left){
            mid = (right+left) / 2;
            if (_interpolants[left].interval().contains(t)){
                return left;
            }
            else if (_interpolants[mid].interval().contains(t)){
                return mid;
            }
            else if ( _interpolants[left].interval().start()*d < t*d && t*d < _interpolants[mid].interval().end()*d){
                left++;
                right = mid-1;
            }
            else{
                left = mid+1;
            }
        }
        //now right == left
        return right;
    }

    inline bool _can_replace_last_with(const LocalInterpolator<T, N>& other) const {
        //ideally we should check that the vectors match in the boundary, but this might be too expensive, so the user should check it if necessary.
        return _interpolants.back().interval().is_point() && other.interval().start_bdr() == 1 && (_dir + other.interval().dir() != 0);
    }

    std::vector<LocalInterpolator<T, N>> _interpolants;
    mutable vec<T, N> _q;
    int _dir = 0;
};




template<typename T, int N, class Derived>
inline void DerivedState<T, N, Derived>::adjust(const T& h_abs, const T& dir, const vec<T, N>& diff){
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

template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const T& t, const vec<T, N>& q) : _q_old(q), _interval(t), _max_interval(t){}

template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state) : _coef_mat(coef_matrix), _q_old(old_state.vector()), _theta(coef_matrix.cols()), _interval(old_state.t(), state.t()), _max_interval(_interval){}

template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr) : _coef_mat(coef_matrix), _q_old(old_state.vector()), _theta(coef_matrix.cols()), _interval(old_state.t(), state.t(), left_bdr, right_bdr), _max_interval(_interval){}

template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_matrix, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr, const std::array<T, 2>& range) : _coef_mat(coef_matrix), _q_old(old_state.vector()), _theta(coef_matrix.cols()), _interval(range[0], range[1], left_bdr, right_bdr), _max_interval(old_state.t(), state.t(), left_bdr, right_bdr){

    if (sgn(state.t()-old_state.t()) * _interval.dir() <= 0){
        throw std::runtime_error("Interpolation range must be non-zero");
    }
    else if (!_max_interval.contains(range[0]) || !_max_interval.contains(range[1])){
        throw std::runtime_error("Interpolation range invalid");
    }
}

template<typename T, int N>
int LocalInterpolator<T, N>::order() const{
    return _coef_mat.cols();
}

template<typename T, int N>
inline const Interval<T>& LocalInterpolator<T, N>::interval() const{
    return _interval;
}

template<typename T, int N>
void LocalInterpolator<T, N>::call(vec<T, N>& result, const T& t) const{
    if (is_out_of_bounds(t)){
        throw std::runtime_error("Interpolation error");
    }
    const T& h = _max_interval.h();
    T x = (t-_max_interval.start())/h;
    for (int i=0; i<order(); i++){
        _theta[i] = pow(x, i+1);
    }
    result = _q_old + (h*_coef_mat.matrix() * _theta.matrix()).array();
}

template<typename T, int N>
inline bool LocalInterpolator<T, N>::is_out_of_bounds(const T& t) const{
    return !_interval.contains(t);
}



#endif