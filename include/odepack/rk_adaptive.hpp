#ifndef ADAPTIVE_RK_HPP
#define ADAPTIVE_RK_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method


#include "solver_impl.hpp"

// Forward declarations
template<typename T, size_t Nstages>
T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size);

template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder>
class RungeKuttaBase : public DerivedSolver<T, N, Derived>{

    using Base = DerivedSolver<T, N, Derived>;

public:

    using Atype = Array2D<T, Nstages, Nstages>;
    using Btype = Array1D<T, Nstages>;
    using Ctype = Array1D<T, Nstages>;

    static constexpr size_t ERR_EST_ORDER = Derived::ERR_EST_ORDER;
    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1);
    static constexpr bool IS_IMPLICIT = false;

    void adapt_impl(State<T, N>& res);

    void reset() override{
        Base::reset();
        _K_true.set(0);
        _mat_is_set = false;
    }

    inline void re_adjust(){}

protected:

    Atype A = Derived::Amatrix();
    Btype B = Derived::Bmatrix();
    Ctype C = Derived::Cmatrix();

    RungeKuttaBase(SOLVER_CONSTRUCTOR(T, N), size_t Krows) : Base(name, ARGS), _K_true(Krows, q0.size()), _df_tmp(q0.size()), _scale_tmp(q0.size()), _error_tmp(q0.size()), _coef_mat(q0.size(), Derived::INTERP_ORDER) {}

    DEFAULT_RULE_OF_FOUR(RungeKuttaBase)

    inline T _estimate_error_norm(const T* K, const T* scale, T h) const {
        return static_cast<const Derived*>(this)->_estimate_error_norm(K, scale, h);
    }

    inline void _set_coef_matrix_impl() const{ //override
        static_cast<const Derived*>(this)->_set_coef_matrix_impl();
    }

    inline void _set_coef_matrix() const{
        if (!this->_mat_is_set){
            this->_set_coef_matrix_impl();
            this->_mat_is_set = true;
        }
        
    }

    void _step_impl(State<T, N>& result, const T& h);
    
    mutable Array2D<T, 0, N>    _K_true;
    mutable Array1D<T, N>       _df_tmp;
    mutable Array1D<T, N>       _scale_tmp;
    mutable Array1D<T, N>       _error_tmp;
    mutable Array2D<T, N, 0>    _coef_mat;
    mutable bool                _mat_is_set = false;
};


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder>
class StandardRungeKutta : public RungeKuttaBase<Derived, T, N, Nstages, Norder>{

    using Base = RungeKuttaBase<Derived, T, N, Nstages, Norder>;

    friend Base;

public:

    using Etype = Array1D<T, Nstages+1>;
    using Ptype = Array2D<T, Nstages+1, 0>;

    Etype E = Derived::Ematrix();
    Ptype P = Derived::Pmatrix();

    inline void interp_impl(T* result, const T& t) const{
        this->_set_coef_matrix();
        return coef_mat_interp(result, t, this->old_state().t, this->current_state().t, this->old_state().vector.data(), this->current_state().vector.data(), this->_coef_mat.data(), Derived::INTERP_ORDER, this->Nsys());
    }

    inline std::unique_ptr<Interpolator<T, N>> state_interpolator(int bdr1, int bdr2) const{
        this->_set_coef_matrix();
        return std::unique_ptr<Interpolator<T, N>>(new StandardLocalInterpolator<T, N>(this->_coef_mat, this->old_state().t, this->t(), this->old_state().vector, this->current_state().vector, bdr1, bdr2));
    }

protected:
    

    StandardRungeKutta(SOLVER_CONSTRUCTOR(T, N)) : Base(name, ARGS, Nstages+1) {}

    DEFAULT_RULE_OF_FOUR(StandardRungeKutta)

    void _set_coef_matrix_impl() const{
        for (size_t i=0; i<this->Nsys(); i++){
            for (size_t j=0; j<Derived::INTERP_ORDER; j++){
                T sum = 0;
                for (size_t k=0; k<Nstages+1; k++){
                    sum += this->_K_true(k, i) * P(k, j);
                }
                this->_coef_mat(i, j) = sum;
            }
        }
    }

private:

    inline T _estimate_error_norm(const T* K, const T* scale, T h) const {
        return _error_norm<T, Nstages>(this->_error_tmp.data(), E.data(), K, h, scale, this->Nsys());
    }

};



template<typename T, size_t N>
class RK45 : public StandardRungeKutta<RK45<T, N>, T, N, 6, 5>{

    static const size_t Norder = 5;
    static const size_t Nstages = 6;
    

    using RKbase = StandardRungeKutta<RK45<T, N>, T, N, Nstages, Norder>;

public:

    static constexpr size_t ERR_EST_ORDER = 4;
    static constexpr size_t INTERP_ORDER = 4;

    RK45(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(RK45);
    
    inline static constexpr typename RKbase::Atype Amatrix();

    inline static constexpr typename RKbase::Btype Bmatrix();

    inline static constexpr typename RKbase::Ctype Cmatrix();

    inline static constexpr typename RKbase::Etype Ematrix();

    inline static constexpr typename RKbase::Ptype Pmatrix();

};




template<typename T, size_t N>
class RK23 : public StandardRungeKutta<RK23<T, N>, T, N, 3, 3> {

    static const size_t Norder = 3;
    static const size_t Nstages = 3;
    

    using RKbase = StandardRungeKutta<RK23<T, N>, T, N, Nstages, Norder>;
    
public:

    static constexpr size_t ERR_EST_ORDER = 2;
    static constexpr size_t INTERP_ORDER = 3;

    RK23(MAIN_DEFAULT_CONSTRUCTOR(T, N));

    DEFAULT_RULE_OF_FOUR(RK23);

    inline static constexpr typename RKbase::Atype Amatrix();

    inline static constexpr typename RKbase::Btype Bmatrix();

    inline static constexpr typename RKbase::Ctype Cmatrix();

    inline static constexpr typename RKbase::Etype Ematrix();

    inline static constexpr typename RKbase::Ptype Pmatrix();

};











//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//--------------------------------IMPLEMENTATIONS-------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------




template<typename T>
inline void adapt_scale(T* scale, const T* q1, const T* q2, T atol, T rtol, size_t size){
    for (size_t i=0; i<size; i++){
        scale[i] = atol + std::max(abs(q1[i]), abs(q2[i]))*rtol;
    }
}

// template<typename T, size_t Nstages>
// T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size) {

//     for (size_t j=0; j<size; j++){
//         T sum = 0;
//         for (size_t i=0; i<Nstages+1; i++){
//             sum += K[i*size+j] * E[i]*h;
//         }
//         tmp[j] = sum/scale[j];
//     }
//     return rms_norm(tmp, size);
// }

template<typename T, size_t Nstages>
T _error_norm(T* tmp, const T* E, const T* K, const T& h, const T* scale, size_t size) {

    // for (size_t j=0; j<size; j++){
    //     T sum = 0;
    //     for (size_t i=0; i<Nstages+1; i++){
    //         sum += K[i*size+j] * E[i]*h;
    //     }
    //     tmp[j] = sum/scale[j];
    // }
    // The implementation below can be vectorized

    std::fill(tmp, tmp+size, 0);
    for (size_t i=0; i<Nstages+1; i++){
        T p = E[i]*h;
        #pragma omp simd
        for (size_t j=0; j<size; j++){
            tmp[j] += (K[i*size+j] * p)/scale[j];
        }
    }
    return rms_norm(tmp, size);
}


template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder>
void RungeKuttaBase<Derived, T, N, Nstages, Norder>::adapt_impl(State<T, N>& res){
    const T& h_min = this->min_step();
    const T& max_step = this->max_step();
    const T& atol = this->atol();
    const T& rtol = this->rtol();
    const State<T, N>& state = this->current_state();

    res.habs = state.habs;
    T h, err_norm, factor, _factor;

    bool step_accepted = false;
    bool step_rejected = false;
    while (!step_accepted){

        h = res.habs * this->direction();
        _step_impl(res, h); //res and K are altered
        adapt_scale(_scale_tmp.data(), state.vector.data(), res.vector.data(), atol, rtol, this->Nsys());
        err_norm = this->_estimate_error_norm(_K_true.data(), _scale_tmp.data(), h);
        _factor = this->SAFETY*pow(err_norm, ERR_EXP);
        if (err_norm < 1){
            factor = (err_norm == 0) ? this->MAX_FACTOR : std::min(this->MAX_FACTOR, _factor);
            if (step_rejected){
                factor = factor < 1 ? factor : 1;
            }
            step_accepted = true;
        }
        else{
            factor = std::max(this->MIN_FACTOR, _factor);
            step_rejected = true;
        }
        if (!resize_step(factor, res.habs, h_min, max_step)){
            break;
        }
    }
}



template<typename Derived, typename T, size_t N, size_t Nstages, size_t Norder>
void RungeKuttaBase<Derived, T, N, Nstages, Norder>::_step_impl(State<T, N>& result, const T& h){
    this->_mat_is_set = false;
    const State<T, N>& state = this->current_state();
    size_t Nsys = this->Nsys();
    T* q = result.vector.data();
    T* K_ = _K_true.data();
    const T* vector = state.vector.data();
    // const T* A_ = A.data();
    const T* B_ = B.data();
    const T* C_ = C.data();
    T* r = _df_tmp.data();

    this->_rhs(K_, state.t, vector);

    #pragma omp simd
    for (size_t j=0; j<Nsys; j++){
        q[j] = vector[j]+B_[0]*K_[j]*h;
    }

    for (size_t s=1; s<Nstages; s++){
        #pragma omp simd
        for (size_t j=0; j<Nsys; j++) {
            r[j] = vector[j];
        }
        for (size_t i=0; i<s; i++){
            T p = A(s, i) * h;
            #pragma omp simd
            for (size_t j=0; j<Nsys; j++){
                r[j] += p * _K_true(i, j);
            }
        }
        this->_rhs(K_+s*Nsys, state.t+C_[s]*h, r);
        T p = B(s) * h;
        #pragma omp simd
        for (size_t j=0; j<Nsys; j++){
            q[j] += p * _K_true(s, j);
        }
    }

    this->_rhs(K_+Nstages*Nsys, state.t+h, q);
    result.t = state.t+h;
}


template<typename T, size_t N>
RK45<T, N>::RK45(MAIN_CONSTRUCTOR(T, N)) : RKbase("RK45", ARGS){}

template<typename T, size_t N>
inline constexpr typename RK45<T, N>::RKbase::Atype RK45<T, N>::Amatrix() {
    return {   T(0),        T(0),        T(0),        T(0),        T(0), T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0), T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0), T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0), T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656), T(0)};
}

template<typename T, size_t N>
inline constexpr typename RK45<T, N>::RKbase::Btype RK45<T, N>::Bmatrix(){
    T q[] = { T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84)};
    typename RKbase::Btype B(q);
    return B;
}

template<typename T, size_t N>
inline constexpr typename RK45<T, N>::RKbase::Ctype RK45<T, N>::Cmatrix(){
    
    T q[] = { T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1)};
    typename RKbase::Ctype C(q);
    return C;
}

template<typename T, size_t N>
inline constexpr typename RK45<T, N>::RKbase::Etype RK45<T, N>::Ematrix() {
    
    T q[] = { T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40)};
    typename RKbase::Etype E(q);
    return E;
}

template<typename T, size_t N>
inline constexpr typename RK45<T, N>::RKbase::Ptype RK45<T, N>::Pmatrix() {
    
    T q[] = {    T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423)};
    typename RKbase::Ptype P(q, Nstages+1, static_cast<size_t>(4));
    return P;
}








template<typename T, size_t N>
RK23<T, N>::RK23(MAIN_CONSTRUCTOR(T, N)) : RKbase("RK23", ARGS){}

template<typename T, size_t N>
inline constexpr typename RK23<T, N>::RKbase::Atype RK23<T, N>::Amatrix() {
    return { T(0),    T(0),    T(0),
            T(1)/T(2), T(0),    T(0),
            T(0),    T(3)/T(4), T(0)};
}

template<typename T, size_t N>
inline constexpr typename RK23<T, N>::RKbase::Btype RK23<T, N>::Bmatrix(){
    
    T q[] = { T(2)/T(9),
            T(1)/T(3),
            T(4)/T(9)};
    typename RKbase::Btype B(q);
    return B;
}

template<typename T, size_t N>
inline constexpr typename RK23<T, N>::RKbase::Ctype RK23<T, N>::Cmatrix(){
    
    T q[] = { T(0),
            T(1)/T(2),
            T(3)/T(4)};
    typename RKbase::Ctype C(q);
    return C;
}

template<typename T, size_t N>
inline constexpr typename RK23<T, N>::RKbase::Etype RK23<T, N>::Ematrix() {
    T q[] = { T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8)};
    typename RKbase::Etype E(q);
    return E;
}

template<typename T, size_t N>
inline constexpr typename RK23<T, N>::RKbase::Ptype RK23<T, N>::Pmatrix() {
    T q[] = { T(1),   -T(4)/T(3),  T(5)/T(9),
        T(0),    T(1),      -T(2)/T(3),
        T(0),    T(4)/T(3), -T(8)/T(9),
        T(0),   -T(1),       T(1)};
    typename RKbase::Ptype P(q, Nstages+1, static_cast<size_t>(3));
    return P;
}

#endif