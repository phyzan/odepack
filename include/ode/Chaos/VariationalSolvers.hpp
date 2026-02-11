#ifndef VARIATIONAL_HPP
#define VARIATIONAL_HPP

#include "../Core/Events.hpp"
#include "../OdeInt.hpp"
#include "../Core/VirtualBase.hpp"

namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================

template<typename T, typename Derived = void>
class NormalizationEvent : public PeriodicEvent<T, GetDerived<NormalizationEvent<T, Derived>, Derived>>{

    using Base = PeriodicEvent<T, GetDerived<NormalizationEvent<T, Derived>, Derived>>;
    friend typename Base::Main;

public:

    NormalizationEvent(const std::string& name, T period);

    const T&    logksi() const;

    T           lyap() const;

    T           delta_s() const;

    size_t      Nsys_main() const;

protected:

    void register_impl();

    void reset_impl(int direction);

    void mask_impl(T* out, const T& t, const T* q) const;

    bool is_masked_impl() const;

private:

    T _logksi = 0;
    T _delta_s = 0;
};


template<typename T, size_t N, typename RhsType, typename JacType>
class VariationalSolver : public PolyWrapper<OdeRichSolver<T, N>>{

    using Base = PolyWrapper<OdeRichSolver<T, N>>;

public:

    VariationalSolver(OdeData<RhsType, JacType> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args = {}, const std::string& method = "RK45");

     const NormalizationEvent<T>& main_event() const;

     const T&                     logksi() const;

     T                            lyap() const;

     T                            t_lyap() const;

     T                            delta_s() const;

};

template<typename T, size_t N, typename RhsType, typename JacType>
class VariationalODE : public ODE<T, N>{

public:

    VariationalODE(OdeData<RhsType, JacType> ode, T t0, const T* q0_, size_t nsys, T period, T rtol, T atol, T min_step=0, T max_step=inf<T>(), T stepsize=0, int dir=1, const std::vector<T>& args = {}, std::vector<const Event<T>*> = {}, const std::string& method = "RK45");

    DEFAULT_RULE_OF_FOUR(VariationalODE);

    ODE<T, N>* clone() const override;

     const std::vector<T>& t_lyap() const;

     const std::vector<T>& lyap() const;

     const std::vector<T>& kicks() const;

    void clear() override;

    void reset() override;

private:

     const NormalizationEvent<T>& _main_event()const;

    void _register_event(size_t event) override;

    size_t _ind = 0;
    std::vector<T> _t_lyap = {};
    std::vector<T> _lyap_array = {};
    std::vector<T> _delta_s_arr = {};

};


template<typename T>
void normalized(T* q, size_t nsys);


} // namespace ode

#endif
