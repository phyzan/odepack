#ifndef DOP853_HPP
#define DOP853_HPP

#include "DOPRI.hpp"

namespace ode{

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T>
struct DOP_COEFS{

    inline static constexpr size_t N_STAGES = 12;
    inline static constexpr size_t N_STAGES_EXT = 16;
    inline static constexpr size_t INTERP_ORDER = 7;
    inline static constexpr int ERR_EST_ORDER = 7;

    using DOP_A = Array2D<T, N_STAGES_EXT, N_STAGES_EXT>;
    using DOP_B = Array1D<T, N_STAGES>;
    using DOP_C = Array1D<T, 16>;
    using DOP_D = Array2D<T, INTERP_ORDER - 3, N_STAGES_EXT>;
    using DOP_E = Array1D<T, N_STAGES+1>;

    static DOP_A make_A();

    static DOP_B make_B();

    static DOP_C make_C();

    static DOP_E make_E3();

    static DOP_E make_E5();

    static DOP_D make_D();

    DOP_A A = make_A();
    DOP_B B = make_B();
    DOP_C C = make_C();
    DOP_E E3 = make_E3();
    DOP_E E5 = make_E5();
    DOP_D D = make_D();

};

template<typename T>
void coef_mat_interp_dop853(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size);


template<typename T, size_t N>
class DOP853LocalInterpolator final: public LocalInterpolator<T, N>{

public:

    DOP853LocalInterpolator() = delete;

    DOP853LocalInterpolator(const T& t, const T* q, size_t nsys);

    DOP853LocalInterpolator(const Array2D<T, N, 0>& coef_mat, T t1, T t2, const T* y1, const T* y2, size_t nsys, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(DOP853LocalInterpolator);

    size_t order() const override;

    DOP853LocalInterpolator<T, N>* clone() const override;

protected:

    Array2D<T, N, 0> _coef_mat;
    size_t _order = 0;

private:

    void _call_impl(T* result, const T& t) const override;

};


template<typename T, size_t N, SolverPolicy SP, typename RhsType = Func<T>, typename JacType = Func<T>>
class DOP853 : public RungeKuttaBase<DOP853<T, N, SP, RhsType, JacType>, T, N, 12, 8, SP, RhsType, JacType>{

public:

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T)) requires (!is_rich<SP>);

    DOP853(MAIN_DEFAULT_CONSTRUCTOR(T), EVENTS events = {}) requires (is_rich<SP>);

    inline std::unique_ptr<Interpolator<T, N>> state_interpolator(int bdr1, int bdr2) const;

private:

    using Base = RungeKuttaBase<DOP853<T, N, SP, RhsType, JacType>, T, N, 12, 8, SP, RhsType, JacType>;
    friend Base::MainSolverType;

    static constexpr const char* name = "DOP853";
    
    static constexpr size_t N_STAGES = 12;
    static constexpr size_t N_ORDER = 8;
    static constexpr size_t N_STAGES_EXTRA = 3;
    static constexpr int    ERR_EST_ORDER = 7;
    static constexpr size_t N_STAGES_EXT = DOP_COEFS<T>::N_STAGES_EXT;
    static constexpr size_t INTERP_ORDER = DOP_COEFS<T>::INTERP_ORDER;
    

    

    using A_EXTRA_TYPE = Array2D<T, N_STAGES_EXTRA, N_STAGES_EXT>;

    using C_EXTRA_TYPE = Array1D<T, N_STAGES_EXTRA>;

    friend Base;
    friend Base::MainSolverType; // So that Base can access specific private methods for static override

    inline void interp_impl(T* result, const T& t) const;

    static typename Base::Atype Amatrix();

    static typename Base::Btype Bmatrix();

    static typename Base::Ctype Cmatrix();

    static A_EXTRA_TYPE Amatrix_extra();

    static C_EXTRA_TYPE Cmatrix_extra();

    A_EXTRA_TYPE A_EXTRA = Amatrix_extra();

    C_EXTRA_TYPE C_EXTRA = Cmatrix_extra();

    typename DOP_COEFS<T>::DOP_D D = DOP_COEFS<T>::make_D();

    typename DOP_COEFS<T>::DOP_E E3 = DOP_COEFS<T>::make_E3();

    typename DOP_COEFS<T>::DOP_E E5 = DOP_COEFS<T>::make_E5();

    void set_coef_matrix_impl() const;

    T estimate_error_norm(const T* K, const T* scale, T h) const;

};


// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

// DOP_COEFS implementations
template<typename T>
typename DOP_COEFS<T>::DOP_A DOP_COEFS<T>::make_A(){
    DOP_A res(N_STAGES_EXT, N_STAGES_EXT);
    res.set(0);

    // Stage 1
    res(1, 0) = 5.26001519587677318785587544488e-2;

    // Stage 2
    res(2, 0) = 1.97250569845378994544595329183e-2;
    res(2, 1) = 5.91751709536136983633785987549e-2;

    // Stage 3
    res(3, 0) = 2.95875854768068491816892993775e-2;
    res(3, 2) = 8.87627564304205475450678981324e-2;

    // Stage 4
    res(4, 0) = 2.41365134159266685502369798665e-1;
    res(4, 2) = -8.84549479328286085344864962717e-1;
    res(4, 3) = 9.24834003261792003115737966543e-1;

    // Stage 5
    res(5, 0) = T(1)/T(27);
    res(5, 3) = 1.70828608729473871279604482173e-1;
    res(5, 4) = 1.25467687566822425016691814123e-1;

    // Stage 6
    res(6, 0) = T(19)/T(512);
    res(6, 3) = 1.70252211019544039314978060272e-1;
    res(6, 4) = 6.02165389804559606850219397283e-2;
    res(6, 5) = T(-9)/T(512);

    // Stage 7
    res(7, 0) = 3.70920001185047927108779319836e-2;
    res(7, 3) = 1.70383925712239993810214054705e-1;
    res(7, 4) = 1.07262030446373284651809199168e-1;
    res(7, 5) = -1.53194377486244017527936158236e-2;
    res(7, 6) = 8.27378916381402288758473766002e-3;

    // Stage 8
    res(8, 0) = 6.24110958716075717114429577812e-1;
    res(8, 3) = -3.36089262944694129406857109825;
    res(8, 4) = -8.68219346841726006818189891453e-1;
    res(8, 5) = 2.75920996994467083049415600797e1;
    res(8, 6) = 2.01540675504778934086186788979e1;
    res(8, 7) = -4.34898841810699588477366255144e1;

    // Stage 9
    res(9, 0) = 4.77662536438264365890433908527e-1;
    res(9, 3) = -2.48811461997166764192642586468;
    res(9, 4) = -5.90290826836842996371446475743e-1;
    res(9, 5) = 2.12300514481811942347288949897e1;
    res(9, 6) = 1.52792336328824235832596922938e1;
    res(9, 7) = -3.32882109689848629194453265587e1;
    res(9, 8) = -2.03312017085086261358222928593e-2;

    // Stage 10
    res(10, 0) = -9.3714243008598732571704021658e-1;
    res(10, 3) = 5.18637242884406370830023853209;
    res(10, 4) = 1.09143734899672957818500254654;
    res(10, 5) = -8.14978701074692612513997267357;
    res(10, 6) = -1.85200656599969598641566180701e1;
    res(10, 7) = 2.27394870993505042818970056734e1;
    res(10, 8) = 2.49360555267965238987089396762;
    res(10, 9) = -3.0467644718982195003823669022;

    // Stage 11
    res(11, 0) = 2.27331014751653820792359768449;
    res(11, 3) = -1.05344954667372501984066689879e1;
    res(11, 4) = -2.00087205822486249909675718444;
    res(11, 5) = -1.79589318631187989172765950534e1;
    res(11, 6) = 2.79488845294199600508499808837e1;
    res(11, 7) = -2.85899827713502369474065508674;
    res(11, 8) = -8.87285693353062954433549289258;
    res(11, 9) = 1.23605671757943030647266201528e1;
    res(11, 10) = 6.43392746015763530355970484046e-1;

    // Stage 12 (this is B coefficients, but part of extended A matrix)
    res(12, 0) = 5.42937341165687622380535766363e-2;
    res(12, 5) = 4.45031289275240888144113950566;
    res(12, 6) = 1.89151789931450038304281599044;
    res(12, 7) = -5.8012039600105847814672114227;
    res(12, 8) = 3.1116436695781989440891606237e-1;
    res(12, 9) = -1.52160949662516078556178806805e-1;
    res(12, 10) = 2.01365400804030348374776537501e-1;
    res(12, 11) = 4.47106157277725905176885569043e-2;

    // Stage 13
    res(13, 0) = 5.61675022830479523392909219681e-2;
    res(13, 6) = 2.53500210216624811088794765333e-1;
    res(13, 7) = -2.46239037470802489917441475441e-1;
    res(13, 8) = -1.24191423263816360469010140626e-1;
    res(13, 9) = 1.5329179827876569731206322685e-1;
    res(13, 10) = 8.20105229563468988491666602057e-3;
    res(13, 11) = 7.56789766054569976138603589584e-3;
    res(13, 12) = -8.298e-3;

    // Stage 14
    res(14, 0) = 3.18346481635021405060768473261e-2;
    res(14, 5) = 2.83009096723667755288322961402e-2;
    res(14, 6) = 5.35419883074385676223797384372e-2;
    res(14, 7) = -5.49237485713909884646569340306e-2;
    res(14, 10) = -1.08347328697249322858509316994e-4;
    res(14, 11) = 3.82571090835658412954920192323e-4;
    res(14, 12) = -3.40465008687404560802977114492e-4;
    res(14, 13) = 1.41312443674632500278074618366e-1;

    // Stage 15
    res(15, 0) = -4.28896301583791923408573538692e-1;
    res(15, 5) = -4.69762141536116384314449447206;
    res(15, 6) = 7.68342119606259904184240953878;
    res(15, 7) = 4.06898981839711007970213554331;
    res(15, 8) = 3.56727187455281109270669543021e-1;
    res(15, 12) = -1.39902416515901462129418009734e-3;
    res(15, 13) = 2.9475147891527723389556272149;
    res(15, 14) = -9.15095847217987001081870187138;

    return res;
}

template<typename T>
typename DOP_COEFS<T>::DOP_B DOP_COEFS<T>::make_B(){
    // B coefficients are row 12 of the A matrix (the 8th order solution weights)
    DOP_A a = make_A();
    DOP_B res(N_STAGES);
    for(size_t i = 0; i < N_STAGES; ++i){
        res(i) = a(12, i);
    }
    return res;
}

template<typename T>
typename DOP_COEFS<T>::DOP_C DOP_COEFS<T>::make_C(){
    DOP_C res(N_STAGES_EXT);
    res(0) = 0;
    res(1) = 0.526001519587677318785587544488e-01;
    res(2) = 0.789002279381515978178381316732e-01;
    res(3) = 0.118350341907227396726757197510;
    res(4) = 0.281649658092772603273242802490;
    res(5) = T(1)/T(3);
    res(6) = T(1)/T(4);
    res(7) = T(4)/T(13);
    res(8) = T(127)/T(195);
    res(9) = T(3)/T(5);
    res(10) = T(6)/T(7);
    res(11) = 1.0;
    res(12) = 1.0;
    res(13) = 0.1;
    res(14) = 0.2;
    res(15) = T(7)/T(9);
    return res;
}

template<typename T>
typename DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E3(){
    DOP_E res(N_STAGES + 1);
    // E3 = B - B3 (3rd order embedded method)
    // Most values are zero, only differences are specified
    res(0) = 5.42937341165687622380535766363e-2 - 0.244094488188976377952755905512;
    res(1) = 0;
    res(2) = 0;
    res(3) = 0;
    res(4) = 0;
    res(5) = 4.45031289275240888144113950566;
    res(6) = 1.89151789931450038304281599044;
    res(7) = -5.8012039600105847814672114227;
    res(8) = 3.1116436695781989440891606237e-1 - 0.733846688281611857341361741547;
    res(9) = -1.52160949662516078556178806805e-1;
    res(10) = 2.01365400804030348374776537501e-1;
    res(11) = 4.47106157277725905176885569043e-2 - 0.220588235294117647058823529412e-1;
    res(12) = 0;
    return res;
}

template<typename T>
typename DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E5(){
    DOP_E res(N_STAGES + 1);
    res.set(0);
    // E5 = B - B5 (5th order embedded method)
    res(0) = 0.1312004499419488073250102996e-1;
    res(5) = -0.1225156446376204440720569753e+1;
    res(6) = -0.4957589496572501915214079952;
    res(7) = 0.1664377182454986536961530415e+1;
    res(8) = -0.3503288487499736816886487290;
    res(9) = 0.3341791187130174790297318841;
    res(10) = 0.8192320648511571246570742613e-1;
    res(11) = -0.2235530786388629525884427845e-1;
    return res;
}

template<typename T>
typename DOP_COEFS<T>::DOP_D DOP_COEFS<T>::make_D(){
    DOP_D res(INTERP_ORDER - 3, N_STAGES_EXT);
    res.set(0);

    // D[0, :] - first interpolation coefficients
    res(0, 0) = -0.84289382761090128651353491142e+1;
    res(0, 5) = 0.56671495351937776962531783590;
    res(0, 6) = -0.30689499459498916912797304727e+1;
    res(0, 7) = 0.23846676565120698287728149680e+1;
    res(0, 8) = 0.21170345824450282767155149946e+1;
    res(0, 9) = -0.87139158377797299206789907490;
    res(0, 10) = 0.22404374302607882758541771650e+1;
    res(0, 11) = 0.63157877876946881815570249290;
    res(0, 12) = -0.88990336451333310820698117400e-1;
    res(0, 13) = 0.18148505520854727256656404962e+2;
    res(0, 14) = -0.91946323924783554000451984436e+1;
    res(0, 15) = -0.44360363875948939664310572000e+1;

    // D[1, :] - second interpolation coefficients
    res(1, 0) = 0.10427508642579134603413151009e+2;
    res(1, 5) = 0.24228349177525818288430175319e+3;
    res(1, 6) = 0.16520045171727028198505394887e+3;
    res(1, 7) = -0.37454675472269020279518312152e+3;
    res(1, 8) = -0.22113666853125306036270938578e+2;
    res(1, 9) = 0.77334326684722638389603898808e+1;
    res(1, 10) = -0.30674084731089398182061213626e+2;
    res(1, 11) = -0.93321305264302278729567221706e+1;
    res(1, 12) = 0.15697238121770843886131091075e+2;
    res(1, 13) = -0.31139403219565177677282850411e+2;
    res(1, 14) = -0.93529243588444783865713862664e+1;
    res(1, 15) = 0.35816841486394083752465898540e+2;

    // D[2, :] - third interpolation coefficients
    res(2, 0) = 0.19985053242002433820987653617e+2;
    res(2, 5) = -0.38703730874935176555105901742e+3;
    res(2, 6) = -0.18917813819516756882830838328e+3;
    res(2, 7) = 0.52780815920542364900561016686e+3;
    res(2, 8) = -0.11573902539959630126141871134e+2;
    res(2, 9) = 0.68812326946963000169666922661e+1;
    res(2, 10) = -0.10006050966910838403183860980e+1;
    res(2, 11) = 0.77771377980534432092869265740;
    res(2, 12) = -0.27782057523535084065932004339e+1;
    res(2, 13) = -0.60196695231264120758267380846e+2;
    res(2, 14) = 0.84320405506677161018159903784e+2;
    res(2, 15) = 0.11992291136182789328035130030e+2;

    // D[3, :] - fourth interpolation coefficients
    res(3, 0) = -0.25693933462703749003312586129e+2;
    res(3, 5) = -0.15418974869023643374053993627e+3;
    res(3, 6) = -0.23152937917604549567536039109e+3;
    res(3, 7) = 0.35763911791061412378285349910e+3;
    res(3, 8) = 0.93405324183624310003907691704e+2;
    res(3, 9) = -0.37458323136451633156875139351e+2;
    res(3, 10) = 0.10409964950896230045147246184e+3;
    res(3, 11) = 0.29840293426660503123344363579e+2;
    res(3, 12) = -0.43533456590011143754432175058e+2;
    res(3, 13) = 0.96324553959188282948394950600e+2;
    res(3, 14) = -0.39177261675615439165231486172e+2;
    res(3, 15) = -0.14972683625798562581422125276e+3;

    return res;
}

// coef_mat_interp_dop853 implementation
template<typename T>
void coef_mat_interp_dop853(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size){
    // coef_mat dimensions: size x order (7 columns for DOP853)
    // Implements the Horner-like scheme from scipy with alternating θ and (1-θ)
    // y(θ) = y_old + sum of terms with alternating θ and (1-θ) multiplications
    if (t == t1){
        copy_array(result, y1, size);
        return;
    }
    else if (t == t2){
        copy_array(result, y2, size);
        return;
    }

    T h = t2 - t1;
    T x = (t - t1) / h;  // θ
    T one_minus_x = T(1) - x;  // (1-θ)

    // Initialize result with 0 (NOT y_old - that gets added at the end)
    #pragma omp simd
    for (size_t i = 0; i < size; i++){
        result[i] = T(0);
    }

    // Horner-like scheme: process F coefficients from last to first
    // for i, f in enumerate(reversed(self.F)):
    //     y += f
    //     if i % 2 == 0: y *= x
    //     else: y *= (1 - x)

    for (size_t idx = 0; idx < order; idx++){
        size_t i = order - 1 - idx;  // reversed index: 6, 5, 4, 3, 2, 1, 0

        // Add F[i] coefficient
        #pragma omp simd
        for (size_t j = 0; j < size; j++){
            result[j] += coef_mat[j * order + i];
        }

        // Multiply by θ or (1-θ) based on iteration
        if (idx % 2 == 0){
            #pragma omp simd
            for (size_t j = 0; j < size; j++){
                result[j] *= x;
            }
        } else {
            #pragma omp simd
            for (size_t j = 0; j < size; j++){
                result[j] *= one_minus_x;
            }
        }
    }

    // Add y_old at the end
    #pragma omp simd
    for (size_t i = 0; i < size; i++){
        result[i] += y1[i];
    }
}

// DOP853LocalInterpolator implementations
template<typename T, size_t N>
DOP853LocalInterpolator<T, N>::DOP853LocalInterpolator(const T& t, const T* q, size_t nsys) : LocalInterpolator<T, N>(t, q, nsys){}

template<typename T, size_t N>
DOP853LocalInterpolator<T, N>::DOP853LocalInterpolator(const Array2D<T, N, 0>& coef_mat, T t1, T t2, const T* y1, const T* y2, size_t nsys, int left_bdr, int right_bdr)
    : LocalInterpolator<T, N>(t1, t2, y1, y2, nsys, left_bdr, right_bdr), _coef_mat(coef_mat), _order(coef_mat.Ncols()){}

template<typename T, size_t N>
size_t DOP853LocalInterpolator<T, N>::order() const{
    return _order;
}

template<typename T, size_t N>
DOP853LocalInterpolator<T, N>* DOP853LocalInterpolator<T, N>::clone() const{
    return new DOP853LocalInterpolator(*this);
}

template<typename T, size_t N>
void DOP853LocalInterpolator<T, N>::_call_impl(T* result, const T& t) const{
    coef_mat_interp_dop853(result, t, this->_t_min(), this->_t_max(), this->q_start().data(), this->q_end().data(), _coef_mat.data(), order(), this->array_size());
}

// DOP853 implementations
template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
DOP853<T, N, SP, RhsType, JacType>::DOP853(MAIN_CONSTRUCTOR(T)) requires (!is_rich<SP>): Base(ARGS, N_STAGES_EXT) {}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
DOP853<T, N, SP, RhsType, JacType>::DOP853(MAIN_CONSTRUCTOR(T), EVENTS events) requires (is_rich<SP>): Base(ARGS, events, N_STAGES_EXT) {}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline void DOP853<T, N, SP, RhsType, JacType>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return coef_mat_interp_dop853(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, this->_coef_mat.data(), INTERP_ORDER, this->Nsys());
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
inline std::unique_ptr<Interpolator<T, N>> DOP853<T, N, SP, RhsType, JacType>::state_interpolator(int bdr1, int bdr2) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return std::unique_ptr<Interpolator<T, N>>(new DOP853LocalInterpolator<T, N>(this->_coef_mat, this->t_old(), d[0],this->old_state_ptr()+2, d+2, this->Nsys(), bdr1, bdr2));
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
typename DOP853<T, N, SP, RhsType, JacType>::Base::Atype DOP853<T, N, SP, RhsType, JacType>::Amatrix(){
    typename Base::Atype result(N_STAGES, N_STAGES);
    typename DOP_COEFS<T>::DOP_A full_A = DOP_COEFS<T>::make_A();

    for(size_t i = 0; i < N_STAGES; ++i){
        for(size_t j = 0; j < N_STAGES; ++j){
            result(i, j) = full_A(i, j);
        }
    }
    return result;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
typename DOP853<T, N, SP, RhsType, JacType>::Base::Btype DOP853<T, N, SP, RhsType, JacType>::Bmatrix(){
    return DOP_COEFS<T>::make_B();
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
typename DOP853<T, N, SP, RhsType, JacType>::Base::Ctype DOP853<T, N, SP, RhsType, JacType>::Cmatrix(){
    typename Base::Ctype result(N_STAGES);
    auto C = DOP_COEFS<T>::make_C();
    copy_array(result.data(), C.data(), N_STAGES);
    return result;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
typename DOP853<T, N, SP, RhsType, JacType>::A_EXTRA_TYPE DOP853<T, N, SP, RhsType, JacType>::Amatrix_extra(){
    Array2D<T, N_STAGES_EXTRA, DOP_COEFS<T>::N_STAGES_EXT> result(N_STAGES_EXTRA, DOP_COEFS<T>::N_STAGES_EXT);
    auto A = DOP_COEFS<T>::make_A();
    copy_array(result.data(), A.data()+(N_STAGES+1)* DOP_COEFS<T>::N_STAGES_EXT, N_STAGES_EXTRA* DOP_COEFS<T>::N_STAGES_EXT);
    return result;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
typename DOP853<T, N, SP, RhsType, JacType>::C_EXTRA_TYPE DOP853<T, N, SP, RhsType, JacType>::Cmatrix_extra(){
    Array1D<T, N_STAGES_EXTRA> result(N_STAGES_EXTRA);
    auto C = DOP_COEFS<T>::make_C();
    copy_array(result.data(), C.data()+N_STAGES+1, N_STAGES_EXTRA);
    return result;
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
void DOP853<T, N, SP, RhsType, JacType>::set_coef_matrix_impl() const{

    T h = this->stepsize() * this->direction();
    const T* y_old = this->old_state_ptr()+2;
    const T& t_old = this->t_old();
    size_t Nsys = this->Nsys();
    T* K = this->_K_true.data();

    for (size_t s = 0; s < N_STAGES_EXTRA; s++){
        size_t stage_idx = N_STAGES + 1 + s;  // 13, 14, 15

        // Compute dy = sum(K[i] * A_EXTRA[s, i]) * h for i < stage_idx
        for (size_t j = 0; j < Nsys; j++){
            this->_df_tmp(j) = y_old[j];
            for (size_t i = 0; i < stage_idx; i++){
                this->_df_tmp(j) += K[i * Nsys + j] * A_EXTRA(s, i) * h;
            }
        }

        // Evaluate K[stage_idx] = f(t_old + C_EXTRA[s] * h, y_old + dy)
        this->rhs(K + stage_idx * Nsys, t_old + C_EXTRA(s) * h, this->_df_tmp.data());
    }


    const T* f_old = K;  // K[0]
    const T* f_new = K + N_STAGES * Nsys;  // K[12]
    const T* y_new = this->new_state_ptr()+2;

    // F[0] = delta_y
    for (size_t i = 0; i < Nsys; i++){
        this->_coef_mat(i, 0) = y_new[i] - y_old[i];
    }

    // F[1] = h * f_old - delta_y
    for (size_t i = 0; i < Nsys; i++){
        this->_coef_mat(i, 1) = h * f_old[i] - (y_new[i] - y_old[i]);
    }

    // F[2] = 2 * delta_y - h * (f_new + f_old)
    for (size_t i = 0; i < Nsys; i++){
        this->_coef_mat(i, 2) = T(2) * (y_new[i] - y_old[i]) - h * (f_new[i] + f_old[i]);
    }

    // F[3:] = h * D @ K
    // D has shape (4, 16), K has shape (16, Nsys)
    // Result has shape (4, Nsys)
    // _coef_mat[i, 3+j] = h * sum_k(D[j, k] * K[k, i])

    for (size_t j = 0; j < 4; j++){  // 4 rows of D
        for (size_t i = 0; i < Nsys; i++){  // Each equation
            T sum = 0;
            for (size_t k = 0; k < N_STAGES_EXT; k++){  // 16 stages
                sum += D(j, k) * K[k * Nsys + i];
            }
            this->_coef_mat(i, 3 + j) = h * sum;
        }
    }
}

template<typename T, size_t N, SolverPolicy SP, typename RhsType, typename JacType>
T DOP853<T, N, SP, RhsType, JacType>::estimate_error_norm(const T* K, const T* scale, T h) const{
    // DOP853 uses a combination of 3rd and 5th order error estimates
    // err5 = K.T @ E5 / scale
    // err3 = K.T @ E3 / scale
    // error_norm = |h| * ||err5||_2 / sqrt(||err5||_2^2 + 0.01 * ||err3||_2^2) * sqrt(n)

    size_t Nsys = this->Nsys();
    T err5_norm_2 = 0;
    T err3_norm_2 = 0;

    // Compute err5 = sum(K[i] * E5[i]) / scale for each equation
    for (size_t j = 0; j < Nsys; j++){
        T err5 = 0;
        T err3 = 0;
        for (size_t i = 0; i < N_STAGES + 1; i++){  // 13 elements in E5 and E3
            err5 += K[i * Nsys + j] * E5(i);
            err3 += K[i * Nsys + j] * E3(i);
        }
        err5 /= scale[j];
        err3 /= scale[j];

        err5_norm_2 += err5 * err5;
        err3_norm_2 += err3 * err3;
    }

    // Handle special case
    if (err5_norm_2 == 0 && err3_norm_2 == 0){
        return T(0);
    }

    T denom = err5_norm_2 + T(1)/100 * err3_norm_2;
    return abs(h) * err5_norm_2 / sqrt(denom * Nsys);
}

} // namespace ode

#endif
