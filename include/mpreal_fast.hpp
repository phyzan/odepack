#ifndef MPREAL_FAST_H
#define MPREAL_FAST_H

#include <mpfr.h>
#include <type_traits>
#include <utility>
#include <limits>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <thread>


namespace mpfr{

#define MP_V_INLINE [[nodiscard]] inline __attribute__((always_inline))
#define MP_INLINE inline __attribute__((always_inline))


#if defined(__GNUC__)
  #define MPREAL_PERMISSIVE_EXPR __extension__
#else
  #define MPREAL_PERMISSIVE_EXPR
#endif

#define THIS static_cast<copy_const_t<std::remove_reference_t<decltype(*this)>, Derived>*>(this)

template<typename From, typename To>
using copy_const_t = std::conditional_t<std::is_const_v<From>, const To, To>;

class mpreal;
template<typename E> struct Expr;
template<typename Derived, typename L, typename R> class BinaryOperator;
template<typename L, typename R> struct AddExpr;
template<typename L, typename R> struct SubExpr;
template<typename L, typename R> struct MulExpr;
template<typename L, typename R> struct DivExpr;
template<typename L, typename R> struct PowExpr;
template<typename E> struct NegExpr;

struct ExprBase{

protected:

    static std::vector<mpreal*> aux_ptrs;
    static std::shared_mutex aux_mutex;

    static void append_ptr(mpreal* ptr){
        std::unique_lock lock(aux_mutex);
        aux_ptrs.push_back(ptr);
    }

public:

    template<typename Func>
    static void for_each_aux(Func&& fn) {
        std::shared_lock lock(aux_mutex);
        for (auto* p : aux_ptrs) {fn(p);};
    }
};

template<typename Derived>
struct Expr : public ExprBase {
    using is_operator_tag = std::false_type;
    using is_expr_tag = std::true_type;

    MP_V_INLINE const mpreal& compute(mp_rnd_t rnd) const {
        return THIS->compute(rnd);
    }
};


template<typename Derived>
class ExprOperator : public Expr<Derived> {

public:
    using Base = Expr<Derived>;
    using is_operator_tag = std::true_type;  // tag for expression detection

    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd) const{
        THIS->assign_to(dest, rnd);
    }

    MP_V_INLINE const mpreal& compute(mp_rnd_t rnd = mpfr_get_default_rounding_mode()) const;

protected:

    static thread_local mpreal* mp_aux;

};




// ===================================================================================
// ============================== Check if type is Expr ==============================
// ===================================================================================

template<typename T, typename = void>
struct is_expr_helper : std::false_type {};

template<typename T>
struct is_expr_helper<T, std::void_t<typename std::decay_t<T>::is_expr_tag>> : std::true_type {};

template<typename T>
constexpr bool is_expr_v = is_expr_helper<T>::value;

// ===================================================================================
// ======================= Check if type is Binary operation =========================
// ===================================================================================

template<typename T, typename = void>
struct is_operator_helper : std::false_type {};

template<typename T>
struct is_operator_helper<T, std::void_t<typename std::decay_t<T>::is_operator_tag>> : std::true_type {};

template<typename T>
constexpr bool is_operator_v = is_operator_helper<T>::value;

// ===================================================================================
// ======================= Check if type is mpreal ===================================
// ===================================================================================

template<typename T>
constexpr bool is_mpreal_v = std::is_same_v<std::decay_t<T>, mpreal>;

// ===================================================================================
// ======================= Check if type is MulExpr ==================================
// ===================================================================================

// Forward declaration
template<typename L, typename R> struct MulExpr;

template<typename T>
struct is_mul_expr : std::false_type {};

template<typename L, typename R>
struct is_mul_expr<MulExpr<L, R>> : std::true_type {};

template<typename T>
constexpr bool is_mul_expr_v = is_mul_expr<std::decay_t<T>>::value;

// Check if MulExpr has mpreal operands (for FMA optimization)
template<typename T>
struct is_mpreal_mul : std::false_type {};

template<>
struct is_mpreal_mul<MulExpr<mpreal, mpreal>> : std::true_type {};

template<typename T>
constexpr bool is_mpreal_mul_v = is_mpreal_mul<std::decay_t<T>>::value;

// ===================================================================================
// ====================== Expression Storage Type Selector ===========================
// ===================================================================================
// For mpreal: store by const reference (fast, no copy)
// For sub-expressions: store by value (they're temporaries, will be moved)
// For arithmetic types: store by value (they're cheap to copy)

template<typename T>
struct expr_storage {
    // Default: store by value (for sub-expressions and arithmetic types)
    using type = std::decay_t<T>;
};

template<>
struct expr_storage<mpreal> {
    using type = const mpreal&;
};

template<>
struct expr_storage<const mpreal> {
    using type = const mpreal&;
};

template<>
struct expr_storage<mpreal&> {
    using type = const mpreal&;
};

template<>
struct expr_storage<const mpreal&> {
    using type = const mpreal&;
};

template<typename T>
using expr_storage_t = typename expr_storage<T>::type;

// ============================================================================
// AddExpr
// ============================================================================

class mpreal : public Expr<mpreal>{

public:
    // =========================== Constructors ===========================


    // Static methods for global precision and rounding mode control

    MP_V_INLINE static mp_rnd_t get_default_rnd() noexcept { return mpfr_get_default_rounding_mode(); }
    MP_V_INLINE static mp_prec_t get_default_prec() noexcept { return mpfr_get_default_prec(); }
    MP_INLINE static void set_default_prec(mp_prec_t prec) {
        mpfr_set_default_prec(prec);
        ExprBase::for_each_aux([prec](mpreal* p) { p->set_prec(prec, MPFR_RNDN); });
    }

    MP_V_INLINE static mp_exp_t get_emin() { return mpfr_get_emin(); }
    MP_V_INLINE static mp_exp_t get_emax() { return mpfr_get_emax(); }
    MP_INLINE static void set_emin(mp_exp_t e) { mpfr_set_emin(e); }
    MP_INLINE static void set_emax(mp_exp_t e) { mpfr_set_emax(e); }

    inline mpreal machine_epsilon(const mpreal& x);

    mpreal() {
        mpfr_init2(mp, get_default_prec());
        mp->_mpfr_exp = __MPFR_EXP_ZERO;
    }

    mpreal(const mpreal& other) {
        mpfr_init2(mp, (mpfr_get_prec)(other.mpfr_srcptr()));
        (mpfr_set)(mp, other.mpfr_srcptr(), get_default_rnd());
    }

    mpreal(mpreal&& other) noexcept {
        mp->_mpfr_d = nullptr;
        mpfr_swap(mp, other.mp);
    }

    mpreal(int v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_si(mp, v, rnd);
    }

    mpreal(unsigned int v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_ui(mp, v, rnd);
    }

    mpreal(long v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_si(mp, v, rnd);
    }

    mpreal(unsigned long v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_ui(mp, v, rnd);
    }

    mpreal(long long v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_si(mp, static_cast<long>(v), rnd);
    }

    mpreal(unsigned long long v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_ui(mp, static_cast<unsigned long>(v), rnd);
    }

    mpreal(float v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_flt(mp, v, rnd);
    }

    mpreal(double v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_d(mp, v, rnd);
    }

    mpreal(long double v, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_ld(mp, v, rnd);
    }

    mpreal(const char* s, mp_prec_t prec = mpreal::get_default_prec(), int base = 10, mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_str(mp, s, base, rnd);
    }

    mpreal(const std::string& s, mp_prec_t prec = mpreal::get_default_prec(), int base = 10, mp_rnd_t rnd = mpreal::get_default_rnd()) {
        mpfr_init2(mp, prec);
        mpfr_set_str(mp, s.c_str(), base, rnd);
    }

    // Constructor from expression template
    template<typename E> requires (is_operator_v<E>)
    mpreal(const E& expr, mp_prec_t prec = mpreal::get_default_prec()) {
        mpfr_init2(mp, prec);
        expr.assign_to(*this, get_default_rnd());
    }

    // =========================== Assignment Operators ===========================

    MP_INLINE mpreal& operator=(const mpreal& other) {
        if (this != &other) {
            mp_prec_t tp = (mpfr_get_prec)(mp);
            mp_prec_t vp = (mpfr_get_prec)(other.mp);
            if (tp != vp) {
                clear();
                mpfr_init2(mp, vp);
            }
            (mpfr_set)(mp, other.mp, get_default_rnd());
        }
        return *this;
    }

    MP_INLINE mpreal& operator=(mpreal&& other) noexcept {
        if (this != &other) {
            mpfr_swap(mp, other.mp);
        }
        return *this;
    }

    MP_INLINE mpreal& operator=(int v) { mpfr_set_si(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(unsigned int v) { mpfr_set_ui(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(long v) { mpfr_set_si(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(unsigned long v) { mpfr_set_ui(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(long long v) { mpfr_set_si(mp, static_cast<long>(v), get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(unsigned long long v) { mpfr_set_ui(mp, static_cast<unsigned long>(v), get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(float v) { mpfr_set_flt(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(double v) { mpfr_set_d(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(long double v) { mpfr_set_ld(mp, v, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(const char* s) { mpfr_set_str(mp, s, 10, get_default_rnd()); return *this; }
    MP_INLINE mpreal& operator=(const std::string& s) { mpfr_set_str(mp, s.c_str(), 10, get_default_rnd()); return *this; }

    template<typename E> requires (is_operator_v<E>)
    MP_INLINE mpreal& operator=(const E& expr) {
        expr.assign_to(*this, get_default_rnd());
        return *this;
    }

    // =========================== Destructor ===========================

    MP_INLINE ~mpreal() {
        if (mp->_mpfr_d != nullptr) {
            mpfr_clear(mp);
        }
    }

    // =========================== Accessors ===========================

    MP_V_INLINE ::mpfr_ptr mpfr_ptr() { return mp; }
    MP_V_INLINE ::mpfr_srcptr mpfr_ptr() const { return mp; }
    MP_V_INLINE ::mpfr_srcptr mpfr_srcptr() const { return mp; }

    MP_V_INLINE mp_prec_t get_prec() const { return (mpfr_get_prec)(mp); }
    MP_V_INLINE mp_prec_t getPrecision() const { return (mpfr_get_prec)(mp); }
    MP_INLINE void set_prec(mp_prec_t prec, mp_rnd_t rnd_mode) { mpfr_prec_round(mpfr_ptr(), prec, rnd_mode); }

    // =========================== Conversion Operators ===========================

    MP_V_INLINE explicit operator double() const { return mpfr_get_d(mp, get_default_rnd()); }
    MP_V_INLINE explicit operator float() const { return mpfr_get_flt(mp, get_default_rnd()); }
    MP_V_INLINE explicit operator long double() const { return mpfr_get_ld(mp, get_default_rnd()); }
    MP_V_INLINE explicit operator long() const { return mpfr_get_si(mp, get_default_rnd()); }
    MP_V_INLINE explicit operator unsigned long() const { return mpfr_get_ui(mp, get_default_rnd()); }
    MP_V_INLINE explicit operator long long() const { return static_cast<long long>(mpfr_get_si(mp, get_default_rnd())); }
    MP_V_INLINE explicit operator unsigned long long() const { return static_cast<unsigned long long>(mpfr_get_ui(mp, get_default_rnd())); }
    MP_V_INLINE explicit operator int() const { return static_cast<int>(mpfr_get_si(mp, get_default_rnd())); }
    MP_V_INLINE explicit operator unsigned int() const { return static_cast<unsigned int>(mpfr_get_ui(mp, get_default_rnd())); }

    MP_V_INLINE double toDouble() const { return mpfr_get_d(mp, get_default_rnd()); }
    MP_V_INLINE long toLong() const { return mpfr_get_si(mp, get_default_rnd()); }

    std::string toString(int digits = 0, int base = 10, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if (digits == 0) {
            digits = static_cast<int>(static_cast<double>((mpfr_get_prec)(mp)) * 0.301029995664); // log10(2)
        }
        char* s = nullptr;
        mpfr_exp_t exp;
        s = mpfr_get_str(nullptr, &exp, base, digits, mp, rnd);
        std::string result(s);
        mpfr_free_str(s);
        if (!result.empty() && result[0] != '-' && result[0] != '@') {
            result.insert(1, ".");
            result += "e" + std::to_string(exp - 1);
        } else if (!result.empty() && result[0] == '-') {
            result.insert(2, ".");
            result += "e" + std::to_string(exp - 1);
        }
        return result;
    }

    // =========================== Queries ===========================

    MP_V_INLINE bool isNan() const { return (mpfr_nan_p)(mp) != 0; }
    MP_V_INLINE bool isInf() const { return (mpfr_inf_p)(mp) != 0; }
    MP_V_INLINE bool isFinite() const { return (mpfr_number_p)(mp) != 0; }
    MP_V_INLINE bool isZero() const { return (mpfr_zero_p)(mp) != 0; }
    MP_V_INLINE bool isRegular() const { return (mpfr_regular_p)(mp) != 0; }
    MP_V_INLINE int sign() const { return (mpfr_sgn)(mp); }

    MP_INLINE mpreal& setNan() { mpfr_set_nan(mp); return *this; }
    MP_INLINE mpreal& setInf(int sign = 1) { mpfr_set_inf(mp, sign); return *this; }
    MP_INLINE mpreal& setZero(int sign = 1) { mpfr_set_zero(mp, sign); return *this; }

    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        (mpfr_set)(dest.mp, mp, rnd);
    }

    // Shift operators (multiply/divide by 2^n)
    MP_V_INLINE mpreal operator<<(long n) const {
        mpreal result;
        mpfr_mul_2si(result.mpfr_ptr(), mp, n, get_default_rnd());
        return result;
    }
    
    MP_V_INLINE mpreal operator>>(long n) const {
        mpreal result;
        mpfr_div_2si(result.mpfr_ptr(), mp, n, get_default_rnd());
        return result;
    }

    MP_INLINE mpreal& operator<<=(long n) {
        mpfr_mul_2si(mp, mp, n, get_default_rnd());
        return *this;
    }

    MP_INLINE mpreal& operator>>=(long n) {
        mpfr_div_2si(mp, mp, n, get_default_rnd());
        return *this;
    }

    MP_V_INLINE const mpreal& compute(mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        return *this;
    }

    // Direct evaluation methods for efficient chaining
    MP_INLINE void eval_into(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        (mpfr_set)(dest.mpfr_ptr(), mp, rnd);
    }

    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        mpfr_add(dest.mpfr_ptr(), dest.mpfr_srcptr(), mp, rnd);
    }

    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        mpfr_sub(dest.mpfr_ptr(), dest.mpfr_srcptr(), mp, rnd);
    }

private:
    mpfr_t mp;

    MP_INLINE void clear() {
        if (mp->_mpfr_d != nullptr) {
            mpfr_clear(mp);
        }
    }

    friend mpreal nextabove(const mpreal &x);
    friend mpreal nextbelow(const mpreal &x);
    friend mpreal nexttoward(const mpreal &x, const mpreal &y);
};

template<typename Derived, typename L, typename R>
class BinaryOperator : public ExprOperator<Derived> {
public:
    using Base = ExprOperator<Derived>;

    // L and R are either mpreal, arithmetic types, or other BinaryExprs (nested expressions)

    template<typename LL, typename RR>
    MP_V_INLINE explicit constexpr BinaryOperator(LL&& l, RR&& r) noexcept : lhs(std::forward<LL>(l)), rhs(std::forward<RR>(r)) {}

    // dest = lhs (op) rhs
    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const{
        // As long as all BinaryOperator classes implement static void perform_operation(mpreal& dest, src_left, src_right), where src_left and src_right are either arithmetic types or mpreal, this will work for all algebraic expressions.
        if constexpr (is_operator_v<L> && is_operator_v<R>){
            lhs.assign_to(dest, rnd); // Compute left sub-expression into dest
            Derived::perform_operation(dest, dest, rhs.compute(rnd));
        }else if constexpr (is_operator_v<L>){
            lhs.assign_to(dest, rnd); // Compute left sub-expression into dest
            Derived::perform_operation(dest, dest, rhs);
        }else if constexpr (is_operator_v<R>){
            rhs.assign_to(dest, rnd); // Compute left sub-expression into dest
            Derived::perform_operation(dest, lhs, dest);
        }else{
            Derived::perform_operation(dest, lhs, rhs);
        }
    }
    // Derived classes must implement static void perform_operation(mpreal& dest, src_left, src_right), where src_left and src_right are either arithmetic types or mpreal.

    // Public accessors for FMA optimizations
    MP_V_INLINE const auto& get_lhs() const noexcept { return lhs; }
    MP_V_INLINE const auto& get_rhs() const noexcept { return rhs; }

    // Default eval_into: just calls assign_to
    MP_INLINE void eval_into(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(dest, rnd);
    }

    // Default add_to: evaluate into dest's auxiliary, then add
    // Derived classes should override for efficiency
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(*Base::mp_aux, rnd);
        mpfr_add(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
    }

    // Default sub_from: evaluate into dest's auxiliary, then subtract
    // Derived classes should override for efficiency
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(*Base::mp_aux, rnd);
        mpfr_sub(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
    }

protected:
    // For mpreal: store by const reference (no copy, fast)
    // For sub-expressions: store by value (moved from temporaries)
    // For arithmetic: store by value (cheap)
    expr_storage_t<L> lhs;
    expr_storage_t<R> rhs;

};


template<typename L, typename R>
struct AddExpr : public BinaryOperator<AddExpr<L, R>, L, R> {
    using Base = BinaryOperator<AddExpr<L, R>, L, R>;
    using Base::Base;

    // Specialized assign_to for fused operations
    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        // Pattern: mpreal + MulExpr<mpreal, mpreal> -> FMA
        if constexpr (is_mpreal_v<L> && is_mpreal_mul_v<R>) {
            // dest = lhs + rhs.lhs * rhs.rhs
            mpfr_fma(dest.mpfr_ptr(),
                     this->get_rhs().get_lhs().mpfr_srcptr(),
                     this->get_rhs().get_rhs().mpfr_srcptr(),
                     this->get_lhs().mpfr_srcptr(), rnd);
        }
        // Pattern: MulExpr<mpreal, mpreal> + mpreal -> FMA
        else if constexpr (is_mpreal_mul_v<L> && is_mpreal_v<R>) {
            // dest = lhs.lhs * lhs.rhs + rhs
            mpfr_fma(dest.mpfr_ptr(),
                     this->get_lhs().get_lhs().mpfr_srcptr(),
                     this->get_lhs().get_rhs().mpfr_srcptr(),
                     this->get_rhs().mpfr_srcptr(), rnd);
        }
        // Pattern: MulExpr<mpreal, mpreal> + MulExpr<mpreal, mpreal> -> FMMA
        else if constexpr (is_mpreal_mul_v<L> && is_mpreal_mul_v<R>) {
            // dest = lhs.lhs * lhs.rhs + rhs.lhs * rhs.rhs
            mpfr_fmma(dest.mpfr_ptr(),
                      this->get_lhs().get_lhs().mpfr_srcptr(),
                      this->get_lhs().get_rhs().mpfr_srcptr(),
                      this->get_rhs().get_lhs().mpfr_srcptr(),
                      this->get_rhs().get_rhs().mpfr_srcptr(), rnd);
        }
        // Optimized path: use eval_into + add_to when both are expr/mpreal
        else if constexpr ((is_operator_v<L> || is_mpreal_v<L>) && (is_operator_v<R> || is_mpreal_v<R>)) {
            this->get_lhs().eval_into(dest, rnd);
            this->get_rhs().add_to(dest, rnd);
        }
        // Default: use base class implementation (for arithmetic types)
        else {
            Base::assign_to(dest, rnd);
        }
    }

    // Optimized add_to: dest += (lhs + rhs) = dest + lhs + rhs
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_operator_v<L> || is_mpreal_v<L>) {
            this->get_lhs().add_to(dest, rnd);
        } else {
            AddExpr::perform_operation(dest, dest, this->get_lhs());
        }
        if constexpr (is_operator_v<R> || is_mpreal_v<R>) {
            this->get_rhs().add_to(dest, rnd);
        } else {
            AddExpr::perform_operation(dest, dest, this->get_rhs());
        }
    }

    // Optimized sub_from: dest -= (lhs + rhs) = dest - lhs - rhs
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_operator_v<L> || is_mpreal_v<L>) {
            this->get_lhs().sub_from(dest, rnd);
        } else {
            SubExpr<mpreal, L>::perform_operation(dest, dest, this->get_lhs());
        }
        if constexpr (is_operator_v<R> || is_mpreal_v<R>) {
            this->get_rhs().sub_from(dest, rnd);
        } else {
            SubExpr<mpreal, R>::perform_operation(dest, dest, this->get_rhs());
        }
    }

    // mpreal + mpreal
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, const mpreal& r) {
        mpfr_add(dest.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal + unsigned long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, unsigned long r) {
        mpfr_add_ui(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // unsigned long + mpreal
    MP_INLINE static void perform_operation(mpreal& dest, unsigned long l, const mpreal& r) {
        mpfr_add_ui(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal + long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, long r) {
        mpfr_add_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // long + mpreal
    MP_INLINE static void perform_operation(mpreal& dest, long l, const mpreal& r) {
        mpfr_add_si(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal + int
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, int r) {
        mpfr_add_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // int + mpreal
    MP_INLINE static void perform_operation(mpreal& dest, int l, const mpreal& r) {
        mpfr_add_si(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal + double
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, double r) {
        mpfr_add_d(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // double + mpreal
    MP_INLINE static void perform_operation(mpreal& dest, double l, const mpreal& r) {
        mpfr_add_d(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
};

template<typename L, typename R>
struct SubExpr : public BinaryOperator<SubExpr<L, R>, L, R> {
    using Base = BinaryOperator<SubExpr<L, R>, L, R>;
    using Base::Base;

    // Specialized assign_to for fused operations
    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        // Pattern: MulExpr<mpreal, mpreal> - mpreal -> FMS (a*b - c)
        if constexpr (is_mpreal_mul_v<L> && is_mpreal_v<R>) {
            // dest = lhs.lhs * lhs.rhs - rhs
            mpfr_fms(dest.mpfr_ptr(),
                     this->get_lhs().get_lhs().mpfr_srcptr(),
                     this->get_lhs().get_rhs().mpfr_srcptr(),
                     this->get_rhs().mpfr_srcptr(), rnd);
        }
        // Pattern: MulExpr<mpreal, mpreal> - MulExpr<mpreal, mpreal> -> FMMS
        else if constexpr (is_mpreal_mul_v<L> && is_mpreal_mul_v<R>) {
            // dest = lhs.lhs * lhs.rhs - rhs.lhs * rhs.rhs
            mpfr_fmms(dest.mpfr_ptr(),
                      this->get_lhs().get_lhs().mpfr_srcptr(),
                      this->get_lhs().get_rhs().mpfr_srcptr(),
                      this->get_rhs().get_lhs().mpfr_srcptr(),
                      this->get_rhs().get_rhs().mpfr_srcptr(), rnd);
        }
        // Pattern: mpreal - MulExpr<mpreal, mpreal> -> negate FMS
        // a - b*c = -(b*c - a) = -fms(b, c, a)
        else if constexpr (is_mpreal_v<L> && is_mpreal_mul_v<R>) {
            mpfr_fms(dest.mpfr_ptr(),
                     this->get_rhs().get_lhs().mpfr_srcptr(),
                     this->get_rhs().get_rhs().mpfr_srcptr(),
                     this->get_lhs().mpfr_srcptr(), rnd);
            mpfr_neg(dest.mpfr_ptr(), dest.mpfr_srcptr(), rnd);
        }
        // Optimized path: use eval_into + sub_from when both are expr/mpreal
        else if constexpr ((is_operator_v<L> || is_mpreal_v<L>) && (is_operator_v<R> || is_mpreal_v<R>)) {
            this->get_lhs().eval_into(dest, rnd);
            this->get_rhs().sub_from(dest, rnd);
        }
        // Default: use base class implementation (for arithmetic types)
        else {
            Base::assign_to(dest, rnd);
        }
    }

    // Optimized add_to: dest += (lhs - rhs) = dest + lhs - rhs
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_operator_v<L> || is_mpreal_v<L>) {
            this->get_lhs().add_to(dest, rnd);
        } else {
            AddExpr<mpreal, L>::perform_operation(dest, dest, this->get_lhs());
        }
        if constexpr (is_operator_v<R> || is_mpreal_v<R>) {
            this->get_rhs().sub_from(dest, rnd);
        } else {
            AddExpr<mpreal, R>::perform_operation(dest, dest, this->get_rhs());
        }
    }

    // Optimized sub_from: dest -= (lhs - rhs) = dest - lhs + rhs
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_operator_v<L> || is_mpreal_v<L>) {
            this->get_lhs().sub_from(dest, rnd);
        } else {
            SubExpr<mpreal, L>::perform_operation(dest, dest, this->get_lhs());
        }
        if constexpr (is_operator_v<R> || is_mpreal_v<R>) {
            this->get_rhs().add_to(dest, rnd);
        } else {
            SubExpr<mpreal, R>::perform_operation(dest, dest, this->get_rhs());
        }
    }

    // mpreal - mpreal
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, const mpreal& r) {
        mpfr_sub(dest.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal - unsigned long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, unsigned long r) {
        mpfr_sub_ui(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // unsigned long - mpreal
    MP_INLINE static void perform_operation(mpreal& dest, unsigned long l, const mpreal& r) {
        mpfr_ui_sub(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal - long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, long r) {
        mpfr_sub_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // long - mpreal
    MP_INLINE static void perform_operation(mpreal& dest, long l, const mpreal& r) {
        mpfr_si_sub(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal - int
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, int r) {
        mpfr_sub_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // int - mpreal
    MP_INLINE static void perform_operation(mpreal& dest, int l, const mpreal& r) {
        mpfr_si_sub(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal - double
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, double r) {
        mpfr_sub_d(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // double - mpreal
    MP_INLINE static void perform_operation(mpreal& dest, double l, const mpreal& r) {
        mpfr_d_sub(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
};

template<typename L, typename R>
struct MulExpr : public BinaryOperator<MulExpr<L, R>, L, R> {
    using Base = BinaryOperator<MulExpr<L, R>, L, R>;
    using Base::Base;

    // Optimized add_to: dest += lhs * rhs using FMA
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_mpreal_v<L> && is_mpreal_v<R>) {
            // dest = lhs * rhs + dest (FMA)
            mpfr_fma(dest.mpfr_ptr(), this->get_lhs().mpfr_srcptr(),
                     this->get_rhs().mpfr_srcptr(), dest.mpfr_srcptr(), rnd);
        } else {
            // Fall back to default: evaluate then add
            this->assign_to(*Base::mp_aux, rnd);
            mpfr_add(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
        }
    }

    // Optimized sub_from: dest -= lhs * rhs
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_mpreal_v<L> && is_mpreal_v<R>) {
            // dest = dest - lhs*rhs = -(lhs*rhs - dest) = -fms(lhs, rhs, dest)
            mpfr_fms(dest.mpfr_ptr(), this->get_lhs().mpfr_srcptr(),
                     this->get_rhs().mpfr_srcptr(), dest.mpfr_srcptr(), rnd);
            mpfr_neg(dest.mpfr_ptr(), dest.mpfr_srcptr(), rnd);
        } else {
            // Fall back to default: evaluate then subtract
            this->assign_to(*Base::mp_aux, rnd);
            mpfr_sub(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
        }
    }

    // mpreal * mpreal
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, const mpreal& r) {
        mpfr_mul(dest.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal * unsigned long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, unsigned long r) {
        mpfr_mul_ui(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // unsigned long * mpreal
    MP_INLINE static void perform_operation(mpreal& dest, unsigned long l, const mpreal& r) {
        mpfr_mul_ui(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal * long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, long r) {
        mpfr_mul_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // long * mpreal
    MP_INLINE static void perform_operation(mpreal& dest, long l, const mpreal& r) {
        mpfr_mul_si(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal * int
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, int r) {
        mpfr_mul_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // int * mpreal
    MP_INLINE static void perform_operation(mpreal& dest, int l, const mpreal& r) {
        mpfr_mul_si(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
    // mpreal * double
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, double r) {
        mpfr_mul_d(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // double * mpreal
    MP_INLINE static void perform_operation(mpreal& dest, double l, const mpreal& r) {
        mpfr_mul_d(dest.mpfr_ptr(), r.mpfr_srcptr(), l, mpreal::get_default_rnd());
    }
};

template<typename L, typename R>
struct DivExpr : public BinaryOperator<DivExpr<L, R>, L, R> {
    using Base = BinaryOperator<DivExpr<L, R>, L, R>;
    using Base::Base;

    // mpreal / mpreal
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, const mpreal& r) {
        mpfr_div(dest.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal / unsigned long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, unsigned long r) {
        mpfr_div_ui(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // unsigned long / mpreal
    MP_INLINE static void perform_operation(mpreal& dest, unsigned long l, const mpreal& r) {
        mpfr_ui_div(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal / long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, long r) {
        mpfr_div_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // long / mpreal
    MP_INLINE static void perform_operation(mpreal& dest, long l, const mpreal& r) {
        mpfr_si_div(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal / int
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, int r) {
        mpfr_div_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // int / mpreal
    MP_INLINE static void perform_operation(mpreal& dest, int l, const mpreal& r) {
        mpfr_si_div(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal / double
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, double r) {
        mpfr_div_d(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // double / mpreal
    MP_INLINE static void perform_operation(mpreal& dest, double l, const mpreal& r) {
        mpfr_d_div(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
};

template<typename L, typename R>
struct PowExpr : public BinaryOperator<PowExpr<L, R>, L, R> {
    using Base = BinaryOperator<PowExpr<L, R>, L, R>;
    using Base::Base;

    // mpreal ^ mpreal
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, const mpreal& r) {
        mpfr_pow(dest.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal ^ unsigned long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, unsigned long r) {
        mpfr_pow_ui(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // unsigned long ^ mpreal
    MP_INLINE static void perform_operation(mpreal& dest, unsigned long l, const mpreal& r) {
        mpfr_ui_pow(dest.mpfr_ptr(), l, r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal ^ long
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, long r) {
        mpfr_pow_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // long ^ mpreal (no direct MPFR function, use dest as temp)
    MP_INLINE static void perform_operation(mpreal& dest, long l, const mpreal& r) {
        mpreal temp;
        mpfr_set_si(temp.mpfr_ptr(), l, mpreal::get_default_rnd());
        mpfr_pow(dest.mpfr_ptr(), temp.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal ^ int
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, int r) {
        mpfr_pow_si(dest.mpfr_ptr(), l.mpfr_srcptr(), r, mpreal::get_default_rnd());
    }
    // int ^ mpreal (no direct MPFR function)
    MP_INLINE static void perform_operation(mpreal& dest, int l, const mpreal& r) {
        mpreal temp;
        mpfr_set_si(temp.mpfr_ptr(), l, mpreal::get_default_rnd());
        mpfr_pow(dest.mpfr_ptr(), temp.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // mpreal ^ double (no direct MPFR function)
    MP_INLINE static void perform_operation(mpreal& dest, const mpreal& l, double r) {
        mpreal temp;
        mpfr_set_d(temp.mpfr_ptr(), r, mpreal::get_default_rnd());
        mpfr_pow(dest.mpfr_ptr(), l.mpfr_srcptr(), temp.mpfr_srcptr(), mpreal::get_default_rnd());
    }
    // double ^ mpreal (no direct MPFR function)
    MP_INLINE static void perform_operation(mpreal& dest, double l, const mpreal& r) {
        mpreal temp;
        mpfr_set_d(temp.mpfr_ptr(), l, mpreal::get_default_rnd());
        mpfr_pow(dest.mpfr_ptr(), temp.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    }
};



template<typename Derived, typename E>
struct UnaryOperator : public ExprOperator<Derived> {
    using Base = ExprOperator<Derived>;

    expr_storage_t<E> operand;

    template<typename EE>
    requires (!std::is_arithmetic_v<EE>)
    MP_V_INLINE explicit constexpr UnaryOperator(EE&& e) noexcept : operand(std::forward<EE>(e)) {}

    MP_INLINE void assign_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const{
        if constexpr (is_operator_v<E>) {
            operand.assign_to(dest, rnd);
        }
        apply(dest, dest, rnd);
    }

    MP_V_INLINE const mpreal& compute(mp_rnd_t rnd = mpreal::get_default_rnd()) const{
        if constexpr (is_operator_v<E>) {
            apply(*Base::mp_aux, operand.compute(rnd), rnd);
        }else{
            static_assert((std::is_same_v<std::decay_t<E>, mpreal>), "Invalid operand type for unary operator");
            apply(*Base::mp_aux, operand, rnd);
        }
        return Base::mp_aux;
    }

    // =========================== static override ==============================
    MP_INLINE static void apply(mpreal& dest, const mpreal& src, mp_rnd_t rnd){
        Derived::apply(dest, src, rnd);
    }
    // ==========================================================================

    // Default eval_into: just calls assign_to
    MP_INLINE void eval_into(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(dest, rnd);
    }

    // Default add_to: evaluate into auxiliary, then add
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(*Base::mp_aux, rnd);
        mpfr_add(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
    }

    // Default sub_from: evaluate into auxiliary, then subtract
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        static_cast<const Derived*>(this)->assign_to(*Base::mp_aux, rnd);
        mpfr_sub(dest.mpfr_ptr(), dest.mpfr_srcptr(), Base::mp_aux->mpfr_srcptr(), rnd);
    }

};

template<typename E>
struct NegExpr : public UnaryOperator<NegExpr<E>, E> {

    using Base = UnaryOperator<NegExpr<E>, E>;
    using Base::Base;

    MP_INLINE static void apply(mpreal& dest, const mpreal& src, mp_rnd_t rnd){
        mpfr_neg(dest.mpfr_ptr(), src.mpfr_srcptr(), rnd);
    }

    // Optimized eval_into for NegExpr
    MP_INLINE void eval_into(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        if constexpr (is_operator_v<E>) {
            this->operand.eval_into(dest, rnd);
        } else {
            this->operand.eval_into(dest, rnd);
        }
        mpfr_neg(dest.mpfr_ptr(), dest.mpfr_srcptr(), rnd);
    }

    // add_to: dest += (-operand) = dest - operand
    MP_INLINE void add_to(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        this->operand.sub_from(dest, rnd);
    }

    // sub_from: dest -= (-operand) = dest + operand
    MP_INLINE void sub_from(mpreal& dest, mp_rnd_t rnd = mpreal::get_default_rnd()) const {
        this->operand.add_to(dest, rnd);
    }
};


template<typename E>
struct AbsExpr : public UnaryOperator<AbsExpr<E>, E> {
    using Base = UnaryOperator<AbsExpr<E>, E>;
    using Base::Base;

    MP_INLINE static void apply(mpreal& dest, const mpreal& src, mp_rnd_t rnd){
        (mpfr_abs)(dest.mpfr_ptr(), src.mpfr_srcptr(), rnd);
    }
};

template<typename Derived>
thread_local mpreal* ExprOperator<Derived>::mp_aux = [](){
    static thread_local mpreal aux(0);
    ExprBase::append_ptr(&aux);
    return &aux; }();

inline std::vector<mpreal*> ExprBase::aux_ptrs;
inline std::shared_mutex ExprBase::aux_mutex;

template<typename Derived>
MP_V_INLINE const mpreal& ExprOperator<Derived>::compute(mp_rnd_t rnd) const {
    assign_to(*mp_aux, rnd);
    return *mp_aux;
}

// ===================================================================================
// ================================= Type Traits =====================================
// ===================================================================================

template<typename T>
constexpr bool is_arithmetic_v = std::is_arithmetic_v<std::decay_t<T>>;

template<typename T>
constexpr bool is_mpreal_or_expr_v = is_mpreal_v<T> || is_operator_v<T>;

template<typename T>
constexpr bool is_valid_operand_v = is_mpreal_v<T> || is_operator_v<T> || is_arithmetic_v<T>;

// ===================================================================================
// ========================== Expression Factory Helpers =============================
// ===================================================================================

template<typename L, typename R>
MP_V_INLINE auto make_add(L&& l, R&& r) {
    return AddExpr<std::decay_t<L>, std::decay_t<R>>(std::forward<L>(l), std::forward<R>(r));
}

template<typename L, typename R>
MP_V_INLINE auto make_sub(L&& l, R&& r) {
    return SubExpr<std::decay_t<L>, std::decay_t<R>>(std::forward<L>(l), std::forward<R>(r));
}

template<typename L, typename R>
MP_V_INLINE auto make_mul(L&& l, R&& r) {
    return MulExpr<std::decay_t<L>, std::decay_t<R>>(std::forward<L>(l), std::forward<R>(r));
}

template<typename L, typename R>
MP_V_INLINE auto make_div(L&& l, R&& r) {
    return DivExpr<std::decay_t<L>, std::decay_t<R>>(std::forward<L>(l), std::forward<R>(r));
}

template<typename L, typename R>
MP_V_INLINE auto make_pow(L&& l, R&& r) {
    return PowExpr<std::decay_t<L>, std::decay_t<R>>(std::forward<L>(l), std::forward<R>(r));
}

template<typename E>
MP_V_INLINE auto make_neg(E&& e) {
    return NegExpr<std::decay_t<E>>(std::forward<E>(e));
}

template<typename E>
MP_V_INLINE auto make_abs(E&& e) {
    return AbsExpr<std::decay_t<E>>(std::forward<E>(e));
}

// ===================================================================================
// ============================= Binary Operators ====================================
// ===================================================================================

// --------------------------------- operator+ ---------------------------------------

// mpreal + mpreal
MP_V_INLINE auto operator+(const mpreal& l, const mpreal& r) { return make_add(l, r); }

// mpreal + arithmetic
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator+(const mpreal& l, T r) { return make_add(l, r); }

// arithmetic + mpreal
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator+(T l, const mpreal& r) { return make_add(l, r); }

// expr + mpreal
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator+(E&& l, const mpreal& r) { return make_add(std::forward<E>(l), r); }

// mpreal + expr
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator+(const mpreal& l, E&& r) { return make_add(l, std::forward<E>(r)); }

// expr + expr
template<typename E1, typename E2> requires (is_operator_v<E1> && is_operator_v<E2>)
MP_V_INLINE auto operator+(E1&& l, E2&& r) { return make_add(std::forward<E1>(l), std::forward<E2>(r)); }

// expr + arithmetic
template<typename E, typename T> requires (is_operator_v<E> && is_arithmetic_v<T>)
MP_V_INLINE auto operator+(E&& l, T r) { return make_add(std::forward<E>(l), r); }

// arithmetic + expr
template<typename T, typename E> requires (is_arithmetic_v<T> && is_operator_v<E>)
MP_V_INLINE auto operator+(T l, E&& r) { return make_add(l, std::forward<E>(r)); }

// --------------------------------- operator- ---------------------------------------

// mpreal - mpreal
MP_V_INLINE auto operator-(const mpreal& l, const mpreal& r) { return make_sub(l, r); }

// mpreal - arithmetic
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator-(const mpreal& l, T r) { return make_sub(l, r); }

// arithmetic - mpreal
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator-(T l, const mpreal& r) { return make_sub(l, r); }

// expr - mpreal
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator-(E&& l, const mpreal& r) { return make_sub(std::forward<E>(l), r); }

// mpreal - expr
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator-(const mpreal& l, E&& r) { return make_sub(l, std::forward<E>(r)); }

// expr - expr
template<typename E1, typename E2> requires (is_operator_v<E1> && is_operator_v<E2>)
MP_V_INLINE auto operator-(E1&& l, E2&& r) { return make_sub(std::forward<E1>(l), std::forward<E2>(r)); }

// expr - arithmetic
template<typename E, typename T> requires (is_operator_v<E> && is_arithmetic_v<T>)
MP_V_INLINE auto operator-(E&& l, T r) { return make_sub(std::forward<E>(l), r); }

// arithmetic - expr
template<typename T, typename E> requires (is_arithmetic_v<T> && is_operator_v<E>)
MP_V_INLINE auto operator-(T l, E&& r) { return make_sub(l, std::forward<E>(r)); }

// --------------------------------- operator* ---------------------------------------

// mpreal * mpreal
MP_V_INLINE auto operator*(const mpreal& l, const mpreal& r) { return make_mul(l, r); }

// mpreal * arithmetic
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator*(const mpreal& l, T r) { return make_mul(l, r); }

// arithmetic * mpreal
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator*(T l, const mpreal& r) { return make_mul(l, r); }

// expr * mpreal
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator*(E&& l, const mpreal& r) { return make_mul(std::forward<E>(l), r); }

// mpreal * expr
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator*(const mpreal& l, E&& r) { return make_mul(l, std::forward<E>(r)); }

// expr * expr
template<typename E1, typename E2> requires (is_operator_v<E1> && is_operator_v<E2>)
MP_V_INLINE auto operator*(E1&& l, E2&& r) { return make_mul(std::forward<E1>(l), std::forward<E2>(r)); }

// expr * arithmetic
template<typename E, typename T> requires (is_operator_v<E> && is_arithmetic_v<T>)
MP_V_INLINE auto operator*(E&& l, T r) { return make_mul(std::forward<E>(l), r); }

// arithmetic * expr
template<typename T, typename E> requires (is_arithmetic_v<T> && is_operator_v<E>)
MP_V_INLINE auto operator*(T l, E&& r) { return make_mul(l, std::forward<E>(r)); }

// --------------------------------- operator/ ---------------------------------------

// mpreal / mpreal
MP_V_INLINE auto operator/(const mpreal& l, const mpreal& r) { return make_div(l, r); }

// mpreal / arithmetic
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator/(const mpreal& l, T r) { return make_div(l, r); }

// arithmetic / mpreal
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto operator/(T l, const mpreal& r) { return make_div(l, r); }

// expr / mpreal
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator/(E&& l, const mpreal& r) { return make_div(std::forward<E>(l), r); }

// mpreal / expr
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto operator/(const mpreal& l, E&& r) { return make_div(l, std::forward<E>(r)); }

// expr / expr
template<typename E1, typename E2> requires (is_operator_v<E1> && is_operator_v<E2>)
MP_V_INLINE auto operator/(E1&& l, E2&& r) { return make_div(std::forward<E1>(l), std::forward<E2>(r)); }

// expr / arithmetic
template<typename E, typename T> requires (is_operator_v<E> && is_arithmetic_v<T>)
MP_V_INLINE auto operator/(E&& l, T r) { return make_div(std::forward<E>(l), r); }

// arithmetic / expr
template<typename T, typename E> requires (is_arithmetic_v<T> && is_operator_v<E>)
MP_V_INLINE auto operator/(T l, E&& r) { return make_div(l, std::forward<E>(r)); }

// --------------------------------- unary operator- ---------------------------------

template<typename E> requires (is_expr_v<E>)
MP_V_INLINE auto operator-(E&& x) { return make_neg(std::forward<E>(x)); }

// --------------------------------- pow function ------------------------------------

// mpreal ^ mpreal
MP_V_INLINE auto pow(const mpreal& base, const mpreal& exp) { return make_pow(base, exp); }

// mpreal ^ arithmetic
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto pow(const mpreal& base, T exp) { return make_pow(base, exp); }

// arithmetic ^ mpreal
template<typename T> requires (is_arithmetic_v<T>)
MP_V_INLINE auto pow(T base, const mpreal& exp) { return make_pow(base, exp); }

// expr ^ mpreal
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto pow(E&& base, const mpreal& exp) { return make_pow(std::forward<E>(base), exp); }

// mpreal ^ expr
template<typename E> requires (is_operator_v<E>)
MP_V_INLINE auto pow(const mpreal& base, E&& exp) { return make_pow(base, std::forward<E>(exp)); }

// expr ^ expr
template<typename E1, typename E2> requires (is_operator_v<E1> && is_operator_v<E2>)
MP_V_INLINE auto pow(E1&& base, E2&& exp) { return make_pow(std::forward<E1>(base), std::forward<E2>(exp)); }

// expr ^ arithmetic
template<typename E, typename T> requires (is_operator_v<E> && is_arithmetic_v<T>)
MP_V_INLINE auto pow(E&& base, T exp) { return make_pow(std::forward<E>(base), exp); }

// arithmetic ^ expr
template<typename T, typename E> requires (is_arithmetic_v<T> && is_operator_v<E>)
MP_V_INLINE auto pow(T base, E&& exp) { return make_pow(base, std::forward<E>(exp)); }

// ===================================================================================
// ========================= Compound Assignment Operators ===========================
// ===================================================================================

// --------------------------------- operator+= --------------------------------------

// mpreal += mpreal
MP_INLINE mpreal& operator+=(mpreal& l, const mpreal& r) {
    mpfr_add(l.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// mpreal += arithmetic
template<typename T>
requires (is_arithmetic_v<T>)
MP_INLINE mpreal& operator+=(mpreal& l, T r) {
    AddExpr<mpreal, T>::perform_operation(l, l, r);
    return l;
}

// mpreal += expr
template<typename E>
requires (is_operator_v<E>)
MP_INLINE mpreal& operator+=(mpreal& l, const E& r) {
    mpfr_add(l.mpfr_ptr(), l.mpfr_srcptr(), r.compute().mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// --------------------------------- operator-= --------------------------------------

// mpreal -= mpreal
MP_INLINE mpreal& operator-=(mpreal& l, const mpreal& r) {
    mpfr_sub(l.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// mpreal -= arithmetic
template<typename T>
requires (is_arithmetic_v<T>)
MP_INLINE mpreal& operator-=(mpreal& l, T r) {
    SubExpr<mpreal, T>::perform_operation(l, l, r);
    return l;
}

// mpreal -= expr
template<typename E>
requires (is_operator_v<E>)
MP_INLINE mpreal& operator-=(mpreal& l, const E& r) {
    mpfr_sub(l.mpfr_ptr(), l.mpfr_srcptr(), r.compute().mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// --------------------------------- operator*= --------------------------------------

// mpreal *= mpreal
MP_INLINE mpreal& operator*=(mpreal& l, const mpreal& r) {
    mpfr_mul(l.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// mpreal *= arithmetic
template<typename T>
requires (is_arithmetic_v<T>)
MP_INLINE mpreal& operator*=(mpreal& l, T r) {
    MulExpr<mpreal, T>::perform_operation(l, l, r);
    return l;
}

// mpreal *= expr
template<typename E>
requires (is_operator_v<E>)
MP_INLINE mpreal& operator*=(mpreal& l, const E& r) {
    mpfr_mul(l.mpfr_ptr(), l.mpfr_srcptr(), r.compute().mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// --------------------------------- operator/= --------------------------------------

// mpreal /= mpreal
MP_INLINE mpreal& operator/=(mpreal& l, const mpreal& r) {
    mpfr_div(l.mpfr_ptr(), l.mpfr_srcptr(), r.mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// mpreal /= arithmetic
template<typename T>
requires (is_arithmetic_v<T>)
MP_INLINE mpreal& operator/=(mpreal& l, T r) {
    DivExpr<mpreal, T>::perform_operation(l, l, r);
    return l;
}

// mpreal /= expr
template<typename E>
requires (is_operator_v<E>)
MP_INLINE mpreal& operator/=(mpreal& l, const E& r) {
    mpfr_div(l.mpfr_ptr(), l.mpfr_srcptr(), r.compute().mpfr_srcptr(), mpreal::get_default_rnd());
    return l;
}

// ===================================================================================
// ============================ Comparison Operators =================================
// ===================================================================================

// mpreal vs mpreal
MP_V_INLINE bool operator==(const mpreal& l, const mpreal& r) { return mpfr_equal_p(l.mpfr_srcptr(), r.mpfr_srcptr()) != 0; }
MP_V_INLINE bool operator!=(const mpreal& l, const mpreal& r) { return mpfr_equal_p(l.mpfr_srcptr(), r.mpfr_srcptr()) == 0; }
MP_V_INLINE bool operator<(const mpreal& l, const mpreal& r) { return mpfr_less_p(l.mpfr_srcptr(), r.mpfr_srcptr()) != 0; }
MP_V_INLINE bool operator<=(const mpreal& l, const mpreal& r) { return mpfr_lessequal_p(l.mpfr_srcptr(), r.mpfr_srcptr()) != 0; }
MP_V_INLINE bool operator>(const mpreal& l, const mpreal& r) { return mpfr_greater_p(l.mpfr_srcptr(), r.mpfr_srcptr()) != 0; }
MP_V_INLINE bool operator>=(const mpreal& l, const mpreal& r) { return mpfr_greaterequal_p(l.mpfr_srcptr(), r.mpfr_srcptr()) != 0; }

// mpreal vs double
MP_V_INLINE bool operator==(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) == 0; }
MP_V_INLINE bool operator!=(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) != 0; }
MP_V_INLINE bool operator<(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) < 0; }
MP_V_INLINE bool operator<=(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) <= 0; }
MP_V_INLINE bool operator>(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) > 0; }
MP_V_INLINE bool operator>=(const mpreal& l, double r) { return mpfr_cmp_d(l.mpfr_srcptr(), r) >= 0; }

MP_V_INLINE bool operator==(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) == 0; }
MP_V_INLINE bool operator!=(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) != 0; }
MP_V_INLINE bool operator<(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) > 0; }
MP_V_INLINE bool operator<=(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) >= 0; }
MP_V_INLINE bool operator>(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) < 0; }
MP_V_INLINE bool operator>=(double l, const mpreal& r) { return mpfr_cmp_d(r.mpfr_srcptr(), l) <= 0; }

// mpreal vs long
MP_V_INLINE bool operator==(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) == 0; }
MP_V_INLINE bool operator!=(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) != 0; }
MP_V_INLINE bool operator<(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) < 0; }
MP_V_INLINE bool operator<=(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) <= 0; }
MP_V_INLINE bool operator>(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) > 0; }
MP_V_INLINE bool operator>=(const mpreal& l, long r) { return mpfr_cmp_si(l.mpfr_srcptr(), r) >= 0; }

MP_V_INLINE bool operator==(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) == 0; }
MP_V_INLINE bool operator!=(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) != 0; }
MP_V_INLINE bool operator<(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) > 0; }
MP_V_INLINE bool operator<=(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) >= 0; }
MP_V_INLINE bool operator>(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) < 0; }
MP_V_INLINE bool operator>=(long l, const mpreal& r) { return mpfr_cmp_si(r.mpfr_srcptr(), l) <= 0; }

// mpreal vs unsigned long
MP_V_INLINE bool operator==(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) == 0; }
MP_V_INLINE bool operator!=(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) != 0; }
MP_V_INLINE bool operator<(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) < 0; }
MP_V_INLINE bool operator<=(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) <= 0; }
MP_V_INLINE bool operator>(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) > 0; }
MP_V_INLINE bool operator>=(const mpreal& l, unsigned long r) { return mpfr_cmp_ui(l.mpfr_srcptr(), r) >= 0; }

MP_V_INLINE bool operator==(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) == 0; }
MP_V_INLINE bool operator!=(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) != 0; }
MP_V_INLINE bool operator<(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) > 0; }
MP_V_INLINE bool operator<=(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) >= 0; }
MP_V_INLINE bool operator>(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) < 0; }
MP_V_INLINE bool operator>=(unsigned long l, const mpreal& r) { return mpfr_cmp_ui(r.mpfr_srcptr(), l) <= 0; }

// mpreal vs int (use long overloads)
MP_V_INLINE bool operator==(const mpreal& l, int r) { return l == static_cast<long>(r); }
MP_V_INLINE bool operator!=(const mpreal& l, int r) { return l != static_cast<long>(r); }
MP_V_INLINE bool operator<(const mpreal& l, int r) { return l < static_cast<long>(r); }
MP_V_INLINE bool operator<=(const mpreal& l, int r) { return l <= static_cast<long>(r); }
MP_V_INLINE bool operator>(const mpreal& l, int r) { return l > static_cast<long>(r); }
MP_V_INLINE bool operator>=(const mpreal& l, int r) { return l >= static_cast<long>(r); }

MP_V_INLINE bool operator==(int l, const mpreal& r) { return static_cast<long>(l) == r; }
MP_V_INLINE bool operator!=(int l, const mpreal& r) { return static_cast<long>(l) != r; }
MP_V_INLINE bool operator<(int l, const mpreal& r) { return static_cast<long>(l) < r; }
MP_V_INLINE bool operator<=(int l, const mpreal& r) { return static_cast<long>(l) <= r; }
MP_V_INLINE bool operator>(int l, const mpreal& r) { return static_cast<long>(l) > r; }
MP_V_INLINE bool operator>=(int l, const mpreal& r) { return static_cast<long>(l) >= r; }

// ===================================================================================
// ================================ Math Functions ===================================
// ===================================================================================

template<typename E>
requires (is_expr_v<E>)
MP_V_INLINE auto abs(E&& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal res;
    (mpfr_abs)(res.mpfr_ptr(), x.compute().mpfr_srcptr(), rnd);
    return res;
}



MP_V_INLINE mpreal sqrt(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_sqrt(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal cbrt(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_cbrt(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal exp(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_exp(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal exp2(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_exp2(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal exp10(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_exp10(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal expm1(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_expm1(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal log(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_log(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal log2(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_log2(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal log10(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_log10(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal log1p(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_log1p(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal sin(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_sin(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal cos(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_cos(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal tan(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_tan(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal asin(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_asin(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal acos(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_acos(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal atan(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_atan(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal atan2(const mpreal& y, const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_atan2(result.mpfr_ptr(), y.mpfr_srcptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal sinh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_sinh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal cosh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_cosh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal tanh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_tanh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal asinh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_asinh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal acosh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_acosh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal atanh(const mpreal& x, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_atanh(result.mpfr_ptr(), x.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal floor(const mpreal& x) {
    mpreal result;
    mpfr_floor(result.mpfr_ptr(), x.mpfr_srcptr());
    return result;
}

MP_V_INLINE mpreal ceil(const mpreal& x) {
    mpreal result;
    mpfr_ceil(result.mpfr_ptr(), x.mpfr_srcptr());
    return result;
}

MP_V_INLINE mpreal trunc(const mpreal& x) {
    mpreal result;
    mpfr_trunc(result.mpfr_ptr(), x.mpfr_srcptr());
    return result;
}

MP_V_INLINE mpreal round(const mpreal& x) {
    mpreal result;
    mpfr_round(result.mpfr_ptr(), x.mpfr_srcptr());
    return result;
}

MP_V_INLINE mpreal fmod(const mpreal& x, const mpreal& y, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_fmod(result.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal hypot(const mpreal& x, const mpreal& y, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_hypot(result.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal fma(const mpreal& x, const mpreal& y, const mpreal& z, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_fma(result.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), z.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal fmax(const mpreal& x, const mpreal& y, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_max(result.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd);
    return result;
}

MP_V_INLINE mpreal fmin(const mpreal& x, const mpreal& y, mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result;
    mpfr_min(result.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd);
    return result;
}

// Constants
MP_V_INLINE mpreal const_pi(mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result(0, prec);
    mpfr_const_pi(result.mpfr_ptr(), rnd);
    return result;
}

MP_V_INLINE mpreal const_euler(mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result(0, prec);
    mpfr_const_euler(result.mpfr_ptr(), rnd);
    return result;
}

MP_V_INLINE mpreal const_log2(mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t rnd = mpreal::get_default_rnd()) {
    mpreal result(0, prec);
    mpfr_const_log2(result.mpfr_ptr(), rnd);
    return result;
}

// ===================================================================================
// ================================= I/O Operators ===================================
// ===================================================================================

inline std::ostream& operator<<(std::ostream& os, const mpreal& x) {
    os << x.toString();
    return os;
}

inline std::istream& operator>>(std::istream& is, mpreal& x) {
    std::string s;
    is >> s;
    x = s;
    return is;
}

// ===================================================================================
// ========================== Numeric Limits Helpers =================================
// ===================================================================================

// Convert precision in bits to decimal digits
inline int bits2digits(mp_prec_t b) {
    constexpr double LOG10_2 = 0.30102999566398119;
    return static_cast<int>(static_cast<double>(b) * LOG10_2);
}


inline mpreal nexttoward (const mpreal& x, const mpreal& y)
{
    mpreal a(x);
    mpfr_nexttoward(a.mp,y.mp);
    return a;
}

inline mpreal nextabove  (const mpreal& x)
{
    mpreal a(x);
    mpfr_nextabove(a.mp);
    return a;
}

inline mpreal nextbelow  (const mpreal& x)
{
    mpreal a(x);
    mpfr_nextbelow(a.mp);
    return a;
}

inline mpreal nextafter (const mpreal& x, const mpreal& y)
{
    return nexttoward(x, y);
}

inline mpreal machine_epsilon(const mpreal& x)
{
    /* the smallest eps such that x + eps != x */
    if( x < 0)
    {
        return nextabove(-x) + x;
    }else{
        return nextabove( x) - x;
    }
}

// Machine epsilon: smallest eps such that 1 + eps != 1
// For precision p, this is 2^(1-p)
inline mpreal machine_epsilon(mp_prec_t prec = mpreal::get_default_prec()) {
    mpreal eps(1, prec);
    return eps >> static_cast<long>(prec - 1);  // eps = 2^(1-prec)
}

// Minimum positive normalized value
inline mpreal minval(mp_prec_t prec = mpreal::get_default_prec()) {
    mpreal x(1, prec);
    return x << static_cast<long>(mpfr_get_emin());
}

// Maximum finite value
inline mpreal maxval(mp_prec_t prec = mpreal::get_default_prec()) {
    mpreal x(1, prec);
    mpreal eps = machine_epsilon(prec);
    x -= eps;  // 1 - epsilon = largest value < 1
    return x << static_cast<long>(mpfr_get_emax());
}

// Infinity constant
inline mpreal const_infinity(int sign = 1, mp_prec_t prec = mpreal::get_default_prec()) {
    mpreal x(0, prec);
    mpfr_set_inf(x.mpfr_ptr(), sign);
    return x;
}

// Free-standing query functions
inline bool isnan(const mpreal& x) { return x.isNan(); }
inline bool isinf(const mpreal& x) { return x.isInf(); }
inline bool isfinite(const mpreal& x) { return x.isFinite(); }
inline bool iszero(const mpreal& x) { return x.isZero(); }
inline int sgn(const mpreal& x) { return x.sign(); }

// Swap function
inline void swap(mpreal& a, mpreal& b) noexcept {
    mpfr_swap(a.mpfr_ptr(), b.mpfr_ptr());
}

} // namespace mpfr

// ===================================================================================
// =============================== Numeric Limits ====================================
// ===================================================================================

namespace std
{
    template <>
    inline void swap(mpfr::mpreal& x, mpfr::mpreal& y) noexcept{
        mpfr::swap(x, y);
    }

    template<>
    class numeric_limits<mpfr::mpreal>{
    public:
        static const bool is_specialized    = true;
        static const bool is_signed         = true;
        static const bool is_integer        = false;
        static const bool is_exact          = false;
        static const int  radix             = 2;

        static const bool has_infinity      = true;
        static const bool has_quiet_NaN     = true;
        static const bool has_signaling_NaN = true;

        static const bool is_iec559         = true;        // = IEEE 754
        static const bool is_bounded        = true;
        static const bool is_modulo         = false;
        static const bool traps             = true;
        static const bool tinyness_before   = true;

        static const float_denorm_style has_denorm  = denorm_absent;

        inline static mpfr::mpreal (min)    (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::minval(precision);  }
        inline static mpfr::mpreal (max)    (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::maxval(precision);  }
        inline static mpfr::mpreal lowest   (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return -mpfr::maxval(precision);  }

        // Returns smallest eps such that 1 + eps != 1 (classic machine epsilon)
        inline static mpfr::mpreal epsilon(mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::machine_epsilon(precision); }

        // Returns smallest eps such that x + eps != x (relative machine epsilon)
        inline static mpfr::mpreal epsilon(const mpfr::mpreal& x) {  return mpfr::machine_epsilon(x);  }

        inline static mpfr::mpreal round_error(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            mp_rnd_t r = mpfr::mpreal::get_default_rnd();

            if(r == GMP_RNDN)  {return mpfr::mpreal(0.5, precision);}
            else               {return mpfr::mpreal(1.0, precision);}
        }

        inline static mpfr::mpreal infinity()         { return mpfr::const_infinity();     }
        inline static mpfr::mpreal quiet_NaN()        { return mpfr::mpreal().setNan();    }
        inline static mpfr::mpreal signaling_NaN()    { return mpfr::mpreal().setNan();    }
        inline static mpfr::mpreal denorm_min()       { return (min)();                    }

        // Please note, exponent range is not fixed in MPFR
        static const int min_exponent = MPFR_EMIN_DEFAULT;
        static const int max_exponent = MPFR_EMAX_DEFAULT;
        MPREAL_PERMISSIVE_EXPR static const int min_exponent10 = (int) (MPFR_EMIN_DEFAULT * 0.3010299956639811);
        MPREAL_PERMISSIVE_EXPR static const int max_exponent10 = (int) (MPFR_EMAX_DEFAULT * 0.3010299956639811);

#if MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS

        // Following members should be constant according to standard, but they can be variable in MPFR
        // So we define them as functions here.
        //
        // This is preferable way for std::numeric_limits<mpfr::mpreal> specialization.
        // But it is incompatible with standard std::numeric_limits and might not work with other libraries, e.g. boost.
        // See below for compatible implementation.
        inline static float_round_style round_style()
        {
            mp_rnd_t r = mpfr::mpreal::get_default_rnd();

            switch (r)
            {
            case GMP_RNDN: return round_to_nearest;
            case GMP_RNDZ: return round_toward_zero;
            case GMP_RNDU: return round_toward_infinity;
            case GMP_RNDD: return round_toward_neg_infinity;
            default: return round_indeterminate;
            }
        }

        inline static int digits()                        {    return int(mpfr::mpreal::get_default_prec());    }
        inline static int digits(const mpfr::mpreal& x)   {    return x.getPrecision();                         }

        inline static int digits10(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            return mpfr::bits2digits(precision);
        }

        inline static int digits10(const mpfr::mpreal& x)
        {
            return mpfr::bits2digits(x.getPrecision());
        }

        inline static int max_digits10(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            return digits10(precision);
        }
#else
        // Digits and round_style are NOT constants when it comes to mpreal.
        // If possible, please use functions digits() and round_style() defined above.
        //
        // These (default) values are preserved for compatibility with existing libraries, e.g. boost.
        // Change them accordingly to your application.
        //
        // For example, if you use 256 bits of precision uniformly in your program, then:
        // digits       = 256
        // digits10     = 77
        // max_digits10 = 78
        //
        // Approximate formula for decimal digits is: digits10 = floor(log10(2) * digits). See bits2digits() for more details.

        static const std::float_round_style round_style = round_to_nearest;
        static const int digits       = 53;
        static const int digits10     = 15;
        static const int max_digits10 = 16;
#endif
    };

} // namespace std


#endif // MPREAL_FAST_H