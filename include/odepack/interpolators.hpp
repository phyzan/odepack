#ifndef INTERPOLATORS_HPP
#define INTERPOLATORS_HPP


#include "states.hpp"


template<typename T>
class Interval{

public:

    Interval(const T& a, const T& b, const int& left_bdr=0, const int& right_bdr=0);

    Interval(const T& x) : _a(x), _b(x), _left(1), _right(1), _is_point(true) {}

    Interval() = default;

    DEFAULT_RULE_OF_FOUR(Interval);

    inline bool contains(const T& t) const;

    inline const T& start() const;

    inline const T& end() const;

    inline const T& h() const;

    inline const int& dir() const;

    inline const int& start_bdr() const;

    inline const int& end_bdr() const;

    inline const bool& is_point() const;

    void open_end();

    inline void close_end();

    inline void close_start();

    void adjust_start(const T& new_start);

    void adjust_end(const T& new_end);

    inline bool can_link_with(const Interval<T>& other) const;

    void link_with(Interval<T>& other);
    
    std::string signature(const int& prec = 3) const;

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
class Interpolator{

public:

    virtual ~Interpolator() = default;

    //ACCESSORS

    virtual const Interval<T>&      interval() const = 0;

    virtual const vec<T, N>&        q_start() const = 0;

    virtual const vec<T, N>&        q_end() const = 0;

    virtual const int&              start_bdr() const = 0;

    virtual const int&              end_bdr() const = 0;

    virtual const int&              dir() const = 0;

    virtual const T&                t_start() const = 0;

    virtual const T&                t_end() const = 0;

    virtual bool                    is_out_of_bounds(const T& t) const = 0;

    virtual bool                    can_link_with(const Interpolator<T, N>& other) const = 0;

    virtual Interpolator<T, N>*     clone() const = 0;

    void                            call(vec<T, N>& result, const T& t) const;

    vec<T, N>                       call(const T& t) const;


    //MODIFIERS

    virtual void link_with(Interpolator<T, N>& other) = 0;

    virtual void link_after(Interpolator<T, N>& other) = 0;

    virtual void adjust_start(const T& t_start) = 0;

    virtual void adjust_end(const T& t_end) = 0;

    virtual void close_end() = 0;

    virtual void close_start() = 0;

protected:

    Interpolator() = delete;

    Interpolator(const int& n);

    DEFAULT_RULE_OF_FOUR(Interpolator);

private:

    virtual void _call_impl(vec<T, N>& result, const T& t) const = 0;

    mutable vec<T, N> _q;

};




template<typename T, int N>
class LocalInterpolator : public Interpolator<T, N>{
    //The purpose of this class is to provide a continuous output between the naturally adapted states
    //of an OdeSolver.
    
public:

    LocalInterpolator() = delete;

    LocalInterpolator(const T& t, const vec<T, N>& q);

    LocalInterpolator(const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(LocalInterpolator);

    //ACCESSORS

    inline const Interval<T>&   interval() const override;

    inline const vec<T, N>&     q_start() const override;

    inline const vec<T, N>&     q_end() const override;

    virtual int                 order() const;

    inline const int&           start_bdr() const override;

    inline const int&           end_bdr() const override;

    inline const int&           dir() const override;

    inline const T&             t_start() const override;

    inline const T&             t_end() const override;

    inline bool                 is_out_of_bounds(const T& t) const override;

    inline bool                 can_link_with(const Interpolator<T, N>& other) const override;

    LocalInterpolator<T, N>*    clone() const override;

    //MODIFIERS

    void                        adjust_start(const T& t_start) override;

    void                        adjust_end(const T& t_end) override;

    void                        link_with(Interpolator<T, N>& other) override;

    void                        link_after(Interpolator<T, N>& other) override;

    inline void                 close_end() override;

    inline void                 close_start() override;

protected:

    const T& _t_min() const;

    const T& _t_max() const;

private:

    void _call_impl(vec<T, N>& result, const T& t) const override;

    Interval<T> _interval;
    T _tmin;
    T _tmax;
    vec<T, N> _q_old;
    vec<T, N> _q;

};





template<typename T, int N, typename STATE>
class StandardLocalInterpolator final: public LocalInterpolator<T, N>{
    
public:

    StandardLocalInterpolator() = delete;

    StandardLocalInterpolator(const T& t, const vec<T, N>& q);

    StandardLocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_mat, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(StandardLocalInterpolator);

    int order() const override;

    StandardLocalInterpolator<T, N, STATE>* clone() const override;

private:

    void _call_impl(vec<T, N>& result, const T& t) const override;

    Eigen::Matrix<T, N, -1> _coef_mat; //number of columns is equal to the interpolation order
    mutable vec<T> _theta;//column vector with "interpolation order" number of rows
    int _order = 0;


};



template<typename T, int N, typename INTERPOLATOR=Interpolator<T, N>>
class LinkedInterpolator : public Interpolator<T, N>{

    static constexpr bool _is_void = std::is_same_v<INTERPOLATOR, Interpolator<T, N>>;

public:

    LinkedInterpolator() = delete;

    LinkedInterpolator(const INTERPOLATOR* other);

    LinkedInterpolator(const T& t, const vec<T, N>& q);

    LinkedInterpolator(const LinkedInterpolator& other);

    LinkedInterpolator(LinkedInterpolator&& other) = default;

    LinkedInterpolator& operator=(const LinkedInterpolator& other);

    LinkedInterpolator& operator=(LinkedInterpolator&& other) = default;

    ~LinkedInterpolator();


    //ACCESSORS

    const Interval<T>&                      interval() const override;

    size_t                                  size() const;

    const vec<T, N>&                        q_start() const override;

    const vec<T, N>&                        q_end() const override;

    const int&                              start_bdr() const override;

    const int&                              end_bdr() const override;

    inline const int&                       dir() const override;

    inline const T&                         t_start() const override;

    inline const T&                         t_end() const override;

    inline const INTERPOLATOR&              interpolant(const size_t& i) const;

    inline const INTERPOLATOR&              last_interpolant() const;

    bool                                    is_out_of_bounds(const T& t) const override;

    inline bool                             can_link_with(const Interpolator<T, N>& interpolant) const override;

    LinkedInterpolator<T, N, INTERPOLATOR>* clone() const override;


    //MODIFIERS

    void                                    link_with(Interpolator<T, N>& interpolant) override;

    void                                    link_after(Interpolator<T, N>& interpolant) override;

    inline void                             adjust_start(const T& t_start) override;

    inline void                             adjust_end(const T& t_new) override;

    void                                    expand(const INTERPOLATOR& interpolant);

    inline void                             open_end();

    inline void                             close_end() override;

    inline void                             close_start() override;

private:

    size_t                              _search_index(const T& t) const;

    inline bool                         _can_replace_last_with(const Interpolator<T, N>& other) const;

    void                                _clear();

    void                                _call_impl(vec<T, N>& result, const T& t) const override;

    inline       INTERPOLATOR&          _get(const size_t& i);

    inline const INTERPOLATOR&          _get(const size_t& i) const;

    inline       INTERPOLATOR&          _get_last();

    inline const INTERPOLATOR&          _get_last() const;
    
    inline       INTERPOLATOR&          _get_safe(const size_t& i);

    inline const INTERPOLATOR&          _get_safe(const size_t& i) const;

    inline bool                         _is_good(const Interpolator<T, N>& interpolant) const;

    std::vector<INTERPOLATOR> _interpolants = {};
    std::vector<Interpolator<T, N>*> _interp_ptrs = {};
    mutable vec<T, N> _q;
    int _dir = 0;
    mutable bool _interval_cached = false;
    mutable Interval<T> _interval;

};



template<typename T, int N>
class OdeSolution : public OdeResult<T, N>{

public:

    using EventMap = OdeResult<T, N>::EventMap;

    OdeSolution(const std::vector<T>& t, const std::vector<vec<T, N>>& q, const EventMap& event_map, bool diverges, bool success, double runtime, const std::string& message, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(t, q, event_map, diverges, success, runtime, message), _interpolator(interpolator.clone()) {}

    OdeSolution(const OdeSolution& other) : OdeResult<T, N>(other), _interpolator(other._interpolator->clone()) {}

    OdeSolution(OdeSolution&& other) : OdeResult<T, N>(std::move(other)), _interpolator(other._interpolator){
        other._interpolator = nullptr;
    }

    OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator) : OdeResult<T, N>(std::move(other)), _interpolator(interpolator.clone()) {}

    OdeSolution& operator=(const OdeSolution& other){
        if (&other != this){
            delete _interpolator;
            _interpolator = other._interpolator->clone();
        }
        return *this;
    }

    OdeSolution& operator=(OdeSolution&& other){
        if (&other != this){
            delete _interpolator;
            _interpolator = other._interpolator;
            other._interpolator = nullptr;
        }
        return *this;
    }

    ~OdeSolution(){
        delete _interpolator;
        _interpolator = nullptr;
    }

    inline vec<T, N> operator()(const T& t) const{
        return _interpolator->call(t);
    }

    OdeSolution<T, N>* clone() const override{ return new OdeSolution<T, N>(*this);}

private:

    Interpolator<T, N>* _interpolator;

};


template<typename T, int N>
class _InterpolatorViewer final : public Interpolator<T, N>{

    //internal use only.
    //the user might end up with this, but will only access it via the base class const interface which is safe.

public:

    _InterpolatorViewer(const std::vector<const Interpolator<T, N>*>& interpolators) : Interpolator<T, N>(interpolators.at(0)->q_start().size()), _interpolators(interpolators) {}

    DEFAULT_RULE_OF_FOUR(_InterpolatorViewer);

    const Interval<T>& interval() const override{
        _interval = Interval<T>(this->t_start(), this->t_end(), this->start_bdr(), this->end_bdr());
        return _interval;
    }

    const vec<T, N>& q_start() const override{
        return _interpolators[0]->q_start();
    }

    const vec<T, N>& q_end() const override{
        return _interpolators.back()->q_end();
    }

    const int& start_bdr() const override{
        return _interpolators[0]->start_bdr();
    }

    const int& end_bdr() const override{
        return _interpolators.back()->end_bdr();
    }

    const int& dir() const override{
        return _interpolators.back()->dir();
    }

    const T& t_start() const override{
        return _interpolators[0]->t_start();
    }

    const T& t_end() const override{
        return _interpolators.back()->t_end();
    }

    bool is_out_of_bounds(const T& t) const override{
        return !interval().contains(t);
    }

    bool can_link_with(const Interpolator<T, N>& other) const override{
        const Interpolator<T, N>* r = _interpolators.back();
        bool can_replace = r->interval().is_point() && other.start_bdr() == 1 && (dir() + other.dir() != 0);
        return r->can_link_with(other) || can_replace;
    }

    LinkedInterpolator<T, N>* clone() const override{
        LinkedInterpolator<T, N>* res = new LinkedInterpolator<T, N>(this->t_start(), this->q_start());

        for (size_t i=0; i<_interpolators.size(); i++){
            res->expand(*_interpolators[i]);
        }
        return res;
    }

    void link_with(Interpolator<T, N>& other) override{}

    void link_after(Interpolator<T, N>& other) override{}

    void adjust_end(const T& end) override{}

    void adjust_start(const T& t_start) override{}

    void close_end() override{}

    void close_start() override{}

private:

    void _call_impl(vec<T, N>& result, const T& t) const override{
        const std::vector<const Interpolator<T, N>*>& r = _interpolators;
        size_t left = 0;
        size_t right = r.size()-1;
        size_t mid;
        const int d = dir();
        while (right > left){
            mid = (right+left) / 2;
            if (!r[left]->is_out_of_bounds(t)){
                right = left;
                break;
            }
            else if (!r[mid]->is_out_of_bounds(t)){
                right = mid;
                break;
            }
            else if ( r[left]->t_start()*d < t*d && t*d < r[mid]->t_end()*d){
                left++;
                right = mid-1;
            }
            else{
                left = mid+1;
            }
        }

        //now "right" is the correct index
        
        r[right]->call(result, t);
    }

    std::vector<const Interpolator<T, N>*> _interpolators;
    mutable Interval<T> _interval;

};


//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//--------------------------------IMPLEMENTATIONS-------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------






template<typename T>
Interval<T>::Interval(const T& a, const T& b, const int& left_bdr, const int& right_bdr) : _a(a), _b(b), _h(b-a), _left(sgn(left_bdr)), _right(sgn(right_bdr)), _dir(sgn(b-a)){
    if (a == b){
        throw std::runtime_error("Zero interval not allowed");
    }
}

template<typename T>
inline bool Interval<T>::contains(const T& t) const{
    return (t==_a) ? _left != -1 : ( (t==_b) ? _right != -1 : (_a*_dir < t*_dir) && (t*_dir < _b*_dir));
}


template<typename T>
inline const T& Interval<T>::start() const{
    return _a;
}


template<typename T>
inline const T& Interval<T>::end() const{
    return _b;
}

template<typename T>
inline const T& Interval<T>::h() const{
    return _h;
}

template<typename T>
inline const int& Interval<T>::dir() const{
    return _dir;
}

template<typename T>
inline const int& Interval<T>::start_bdr() const{
    return _left;
}

template<typename T>
inline const int& Interval<T>::end_bdr() const{
    return _right;
}

template<typename T>
inline const bool& Interval<T>::is_point() const{
    return _is_point;
}

template<typename T>
void Interval<T>::open_end(){
    if (_is_point){
        throw std::runtime_error("Cannot open end boundary on a point interval");
    }
    else{
        _right = -1;
    }
}

template<typename T>
inline void Interval<T>::close_end(){
    _right = 1;
}


template<typename T>
inline void Interval<T>::close_start(){
    _left = 1;
}

template<typename T>
inline void Interval<T>::adjust_start(const T& new_start){
    if (sgn(_b-new_start) == _dir){
        _a = new_start;
        _h = _b - _a;
    }
    else{
        throw std::runtime_error("Cannot resize interval");
    }
}

template<typename T>
inline void Interval<T>::adjust_end(const T& new_end){
    if (sgn(new_end-_a) == _dir){
        _b = new_end;
        _h = _b - _a;
    }
    else{
        throw std::runtime_error("Cannot resize interval");
    }
}


template<typename T>
inline bool Interval<T>::can_link_with(const Interval<T>& other) const{
    //boundaries and directions must match.
    return (_b == other._a) && (_right*other._left != 1) && !(_right == 0 && other._left == 0) && (_dir+other._dir != 0);
}


template<typename T>
void Interval<T>::link_with(Interval<T>& other) {
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

template<typename T>
std::string Interval<T>::signature(const int& prec) const{
    if (is_point()){
        return "[" + to_string(_a, prec) + "]";
    }
    else{
        std::string left = (_left == 1) ? "[" : (_left == -1 ? "(" : "|");
        std::string right = (_right == 1) ? "]" : (_right == -1 ? ")" : "|");
        return left + to_string(_a, prec) + ", " + to_string(_b, prec) + right;
    }
}


template<typename T, int N>
Interpolator<T, N>::Interpolator(const int& n) : _q(n) {}

template<typename T, int N>
void Interpolator<T, N>::call(vec<T, N>& result, const T& t) const{
    if (this->is_out_of_bounds(t)){
        throw std::runtime_error("Scalar " + to_string(t) + " is out of bounds in LinkedInterpolator with range " + interval().signature());
    }
    this->_call_impl(result, t);
}

template<typename T, int N>
vec<T, N> Interpolator<T, N>::call(const T& t) const{
    this->call(_q, t);
    return _q;
}


template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const T& t, const vec<T, N>& q) : Interpolator<T, N>(q.size()), _interval(t), _tmin(t), _tmax(t), _q_old(q), _q(q){}


template<typename T, int N>
LocalInterpolator<T, N>::LocalInterpolator(const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr) : Interpolator<T, N>(state.vector().size()), _interval(old_state.t(), state.t(), left_bdr, right_bdr), _tmin(old_state.t()), _tmax(state.t()), _q_old(old_state.exposed_vector()), _q(state.exposed_vector()){}

template<typename T, int N>
inline const Interval<T>& LocalInterpolator<T, N>::interval() const{
    return _interval;
}

template<typename T, int N>
inline const vec<T, N>& LocalInterpolator<T, N>::q_start() const{
    return _q_old;
}

template<typename T, int N>
inline const vec<T, N>& LocalInterpolator<T, N>::q_end() const{
    return _q;
}

template<typename T, int N>
int LocalInterpolator<T, N>::order() const{
    return 1;
}

template<typename T, int N>
inline const int& LocalInterpolator<T, N>::start_bdr() const{
    return _interval.start_bdr();
}

template<typename T, int N>
inline const int& LocalInterpolator<T, N>::end_bdr() const{
    return _interval.end_bdr();
}

template<typename T, int N>
inline const int& LocalInterpolator<T, N>::dir() const{
    return _interval.dir();
}

template<typename T, int N>
inline const T& LocalInterpolator<T, N>::t_start() const{
    return _interval.start();
}

template<typename T, int N>
inline const T& LocalInterpolator<T, N>::t_end() const{
    return _interval.end();
}

template<typename T, int N>
inline bool LocalInterpolator<T, N>::is_out_of_bounds(const T& t) const{
    return !_interval.contains(t);
}

template<typename T, int N>
inline bool LocalInterpolator<T, N>::can_link_with(const Interpolator<T, N>& other) const{
    
    if (const LocalInterpolator<T, N>* p = dynamic_cast<const LocalInterpolator<T, N>*>(&other)){
        return _interval.can_link_with(p->_interval);
    }
    else{
        return (t_end() == other.t_end()) && (end_bdr()*other.start_bdr() != 1) && !(end_bdr() == 0 && other.start_bdr() == 0) && (dir()+other.dir() != 0);
    }
}

template<typename T, int N>
LocalInterpolator<T, N>* LocalInterpolator<T, N>::clone() const{
    return new LocalInterpolator<T, N>(*this);
}

template<typename T, int N>
void LocalInterpolator<T, N>::adjust_start(const T& t_start){
    const int& d = _interval.dir();
    if ( t_start*d < _interval.end()*d && t_start*d >= _tmin*d){
        _interval.adjust_start(t_start);
    }
    else{
        throw std::runtime_error("Cannot adjust interval");
    }
}

template<typename T, int N>
void LocalInterpolator<T, N>::adjust_end(const T& t_end){
    const int& d = _interval.dir();
    if (_interval.start()*d < t_end*d && t_end*d <= _tmax*d){
        _interval.adjust_end(t_end);
    }
    else{
        throw std::runtime_error("Cannot adjust interval");
    }
}

template<typename T, int N>
void LocalInterpolator<T, N>::link_with(Interpolator<T, N>& other){
    if (LocalInterpolator<T, N>* p = dynamic_cast<LocalInterpolator<T, N>*>(&other)){
        _interval.link_with(p->_interval);
    }
    else{
        other.link_after(*this);
    }
}

template<typename T, int N>
void LocalInterpolator<T, N>::link_after(Interpolator<T, N>& other){
    other.link_with(*this);
}

template<typename T, int N>
inline void LocalInterpolator<T, N>::close_end(){
    _interval.close_end();
}

template<typename T, int N>
inline void LocalInterpolator<T, N>::close_start(){
    _interval.close_start();
}

template<typename T, int N>
const T& LocalInterpolator<T, N>::_t_min() const{
    return _tmin;
}

template<typename T, int N>
const T& LocalInterpolator<T, N>::_t_max() const{
    return _tmax;
}

template<typename T, int N>
void LocalInterpolator<T, N>::_call_impl(vec<T, N>& result, const T& t) const{
    result = _q_old + (_q-_q_old)/(_interval.end() - _interval.start()) * (t-_interval.start());
}





template<typename T, int N, typename STATE>
StandardLocalInterpolator<T, N, STATE>::StandardLocalInterpolator(const T& t, const vec<T, N>& q) : LocalInterpolator<T, N>(t, q){}

template<typename T, int N, typename STATE>
StandardLocalInterpolator<T, N, STATE>::StandardLocalInterpolator(const Eigen::Matrix<T, N, -1>& coef_mat, const State<T, N>& old_state, const State<T, N>& state, int left_bdr, int right_bdr) : LocalInterpolator<T, N>(old_state, state, left_bdr, right_bdr), _coef_mat(coef_mat), _theta(coef_mat.cols()), _order(coef_mat.cols()){}

template<typename T, int N, typename STATE>
int StandardLocalInterpolator<T, N, STATE>::order() const{
    return _order;
}

template<typename T, int N, typename STATE>
StandardLocalInterpolator<T, N, STATE>* StandardLocalInterpolator<T, N, STATE>::clone() const{
    return new StandardLocalInterpolator<T, N, STATE>(*this);
}


template<typename T, int N, typename STATE>
void StandardLocalInterpolator<T, N, STATE>::_call_impl(vec<T, N>& result, const T& t) const{
    T h = this->_t_max() - this->_t_min();
    T x = (t-this->_t_min())/h;
    for (int i=0; i<order(); i++){
        _theta[i] = pow(x, i+1);
    }
    result = this->q_start() + (h*_coef_mat.matrix() * _theta.matrix()).array();
}




template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>::LinkedInterpolator(const INTERPOLATOR* interpolant) : Interpolator<T, N>(interpolant->q_start().size()), _q(interpolant->q_start().size()), _dir(interpolant->dir()){
    if constexpr (_is_void){
        _interp_ptrs.push_back(interpolant->clone());
    }
    else{
        _interpolants.push_back(*interpolant);
    }
}



template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>::LinkedInterpolator(const T& t, const vec<T, N>& q) : Interpolator<T, N>(q.size()), _q(q.size()){
    if constexpr (_is_void) {
        _interp_ptrs.push_back(new LocalInterpolator<T, N>(t, q));
    }
    else{
        _interpolants.push_back(INTERPOLATOR(t, q));
    }
}


template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>::LinkedInterpolator(const LinkedInterpolator& other) : Interpolator<T, N>(other.q_start().size()){
    LinkedInterpolator<T, N, INTERPOLATOR>::operator=(other);
}

template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>& LinkedInterpolator<T, N, INTERPOLATOR>::operator=(const LinkedInterpolator& other){
    if (&other == this) return *this;
    Interpolator<T, N>::operator=(other);
    if constexpr (_is_void){
        _clear();
        _interp_ptrs.resize(other._interp_ptrs.size());
        for (size_t i=0; i<_interp_ptrs.size(); i++){
            _interp_ptrs[i] = other._interp_ptrs[i]->clone();
        }
    }
    else{
        _interpolants = other._interpolants;
    }

    _q = other._q;
    _dir = other._dir;
    return *this;
}


template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>::~LinkedInterpolator(){
    if constexpr (_is_void) {
        _clear();
    }
}


template<typename T, int N, typename INTERPOLATOR>
const Interval<T>& LinkedInterpolator<T, N, INTERPOLATOR>::interval() const{
    if (_interval_cached){
        return _interval;
    }
    else{
        _interval = Interval<T>(t_start(), t_end(), start_bdr(), end_bdr());
        _interval_cached = true;
        return _interval;
    }
}

template<typename T, int N, typename INTERPOLATOR>
size_t LinkedInterpolator<T, N, INTERPOLATOR>::size() const{
    if constexpr (_is_void){
        return _interp_ptrs.size();
    }
    else{
        return _interpolants.size();
    }
}


template<typename T, int N, typename INTERPOLATOR>
const vec<T, N>& LinkedInterpolator<T, N, INTERPOLATOR>::q_start() const{
    return _get(0).q_start();
}


template<typename T, int N, typename INTERPOLATOR>
const vec<T, N>& LinkedInterpolator<T, N, INTERPOLATOR>::q_end() const{
    return _get_last().q_end();
}

template<typename T, int N, typename INTERPOLATOR>
const int& LinkedInterpolator<T, N, INTERPOLATOR>::start_bdr() const{
    return _get(0).start_bdr();
}

template<typename T, int N, typename INTERPOLATOR>
const int& LinkedInterpolator<T, N, INTERPOLATOR>::end_bdr() const{
    return _get_last().end_bdr();
}


template<typename T, int N, typename INTERPOLATOR>
inline const int& LinkedInterpolator<T, N, INTERPOLATOR>::dir() const{
    return _dir;
}

template<typename T, int N, typename INTERPOLATOR>
inline const T& LinkedInterpolator<T, N, INTERPOLATOR>::t_start() const{
    return _get(0).t_start();
}

template<typename T, int N, typename INTERPOLATOR>
inline const T& LinkedInterpolator<T, N, INTERPOLATOR>::t_end() const{
    return _get_last().t_end();
}

template<typename T, int N, typename INTERPOLATOR>
inline const INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::interpolant(const size_t& i) const{
    return _get_safe(i);
}

template<typename T, int N, typename INTERPOLATOR>
inline const INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::last_interpolant() const{
    return _get_last();
}


template<typename T, int N, typename INTERPOLATOR>
bool LinkedInterpolator<T, N, INTERPOLATOR>::is_out_of_bounds(const T& t) const {
    if (t == t_start()){
        return _get(0).is_out_of_bounds(t);
    }
    else if (t == t_end()){
        return _get_last().is_out_of_bounds(t);
    }
    else{
        return (t*_dir < t_start()*_dir) || (t*_dir > t_end()*_dir);
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline bool LinkedInterpolator<T, N, INTERPOLATOR>::can_link_with(const Interpolator<T, N>& interpolant) const{
    return _get_last().can_link_with(interpolant) || _can_replace_last_with(interpolant);
}


template<typename T, int N, typename INTERPOLATOR>
LinkedInterpolator<T, N, INTERPOLATOR>* LinkedInterpolator<T, N, INTERPOLATOR>::clone() const{
    return new LinkedInterpolator<T, N, INTERPOLATOR>(*this);
}

template<typename T, int N, typename INTERPOLATOR>
void LinkedInterpolator<T, N, INTERPOLATOR>::link_with(Interpolator<T, N>& interpolant){
    _interval_cached = false;
    _get_last().link_with(interpolant);
}

template<typename T, int N, typename INTERPOLATOR>
void LinkedInterpolator<T, N, INTERPOLATOR>::link_after(Interpolator<T, N>& interpolant){
    _interval_cached = false;
    _get(0).link_after(interpolant);
}

template<typename T, int N, typename INTERPOLATOR>
inline void LinkedInterpolator<T, N, INTERPOLATOR>::adjust_start(const T& t_new) {
    _interval_cached = false;
    _get(0).adjust_start(t_new);
}

template<typename T, int N, typename INTERPOLATOR>
inline void LinkedInterpolator<T, N, INTERPOLATOR>::adjust_end(const T& t_new) {
    _interval_cached = false;
    _get_last().adjust_end(t_new);
}

template<typename T, int N, typename INTERPOLATOR>
void LinkedInterpolator<T, N, INTERPOLATOR>::expand(const INTERPOLATOR& interpolant) {
    _interval_cached = false;
    if (_can_replace_last_with(interpolant)){
        if constexpr (_is_void){
            delete _interp_ptrs.back();
            _interp_ptrs.back() = interpolant.clone();
        }
        else{
            _interpolants.back() = interpolant;
        }
    }
    else if(_get_last().can_link_with(interpolant)){
        if constexpr (_is_void){
            _interp_ptrs.push_back(interpolant.clone());
            _interp_ptrs[size()-2]->link_with(*_interp_ptrs[size()-1]);
        }
        else{
            _interpolants.push_back(interpolant);
            _interpolants[size()-2].link_with(_interpolants[size()-1]);
        }
    }
    else{
        throw std::runtime_error("Invalid interpolant");
    }

    if ((interpolant.dir() != 0) && (_dir == 0)){
        _dir = interpolant.dir();
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline void LinkedInterpolator<T, N, INTERPOLATOR>::open_end(){
    _interval_cached = false;
    _get_last().open_end();
}


template<typename T, int N, typename INTERPOLATOR>
inline void LinkedInterpolator<T, N, INTERPOLATOR>::close_end(){
    _interval_cached = false;
    _get_last().close_end();
}


template<typename T, int N, typename INTERPOLATOR>
inline void LinkedInterpolator<T, N, INTERPOLATOR>::close_start(){
    _interval_cached = false;
    _get(0).close_end();
}


template<typename T, int N, typename INTERPOLATOR>
size_t LinkedInterpolator<T, N, INTERPOLATOR>::_search_index(const T& t) const{
    //no need to check for out of bounds, it must have just been checked before this function call.
    //t must NOT be equal to the _t_end(). This case has already been taken care of.

    size_t left = 0;
    size_t right = size()-1;
    size_t mid;
    const int d = dir();
    while (right > left){
        mid = (right+left) / 2;
        if (!_get(left).is_out_of_bounds(t)){
            return left;
        }
        else if (!_get(mid).is_out_of_bounds(t)){
            return mid;
        }
        else if ( _get(left).t_start()*d < t*d && t*d < _get(mid).t_end()*d){
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

template<typename T, int N, typename INTERPOLATOR>
inline bool LinkedInterpolator<T, N, INTERPOLATOR>::_can_replace_last_with(const Interpolator<T, N>& other) const {
    //ideally we should check that the vectors match in the boundary, but this might be too expensive, so the user should check it if necessary.
    return (_get_last().t_start() == _get_last().t_end()) && other.start_bdr() == 1 && (_dir + other.dir() != 0);
}

template<typename T, int N, typename INTERPOLATOR>
void LinkedInterpolator<T, N, INTERPOLATOR>::_clear() {
    if constexpr (_is_void){
        for (size_t i=0; i<_interp_ptrs.size(); i++){
            delete _interp_ptrs[i];
        }
        _interp_ptrs.clear();
    }
    else{
        _interpolants.clear();
    }
}

template<typename T, int N, typename INTERPOLATOR>
void LinkedInterpolator<T, N, INTERPOLATOR>::_call_impl(vec<T, N>& result, const T& t) const{
    size_t ind = _search_index(t);
    _get(ind).call(result, t);
}


template<typename T, int N, typename INTERPOLATOR>
inline INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get(const size_t& i) {
    if constexpr (_is_void){
        return *_interp_ptrs[i];
    }
    else{
        return _interpolants[i];
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline const INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get(const size_t& i) const{
    if constexpr (_is_void){
        return *_interp_ptrs[i];
    }
    else{
        return _interpolants[i];
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get_last() {
    if constexpr (_is_void){
        return *_interp_ptrs.back();
    }
    else{
        return _interpolants.back();
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline const INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get_last() const {
    if constexpr (_is_void){
        return *_interp_ptrs.back();
    }
    else{
        return _interpolants.back();
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get_safe(const size_t& i){
    if constexpr (_is_void){
        return *_interp_ptrs.at(i);
    }
    else{
        return _interpolants.at(i);
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline const INTERPOLATOR& LinkedInterpolator<T, N, INTERPOLATOR>::_get_safe(const size_t& i) const {
    if constexpr (_is_void){
        return *_interp_ptrs.at(i);
    }
    else{
        return _interpolants.at(i);
    }
}

template<typename T, int N, typename INTERPOLATOR>
inline bool LinkedInterpolator<T, N, INTERPOLATOR>::_is_good(const Interpolator<T, N>& interpolant) const {
    if constexpr (!_is_void) {
        return !(dynamic_cast<const INTERPOLATOR*>(&interpolant));
    } else {
        return true;
    }
}

#endif