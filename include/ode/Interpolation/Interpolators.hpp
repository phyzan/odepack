#ifndef INTERPOLATORS_HPP
#define INTERPOLATORS_HPP

#include "../Tools.hpp"

namespace ode {

template<typename T>
class Interval{

public:

    Interval(const T& a, const T& b, int left_bdr=0, int right_bdr=0);

    Interval(const T& x) : _a(x), _b(x), _left(1), _right(1), _is_point(true) {}

    Interval() = default;

    DEFAULT_RULE_OF_FOUR(Interval);

    // ACCESSORS

     bool         contains(const T& t) const;

     const T&     start() const;

     const T&     end() const;

     const T&     h() const;

     int          dir() const;

     int          start_bdr() const;

     int          end_bdr() const;

     bool         is_point() const;

     bool         can_link_with(const Interval<T>& other) const;

    std::string         signature(int prec = 3) const;

    // MODIFIERS

    void                open_end();

     void         close_end();

     void         close_start();

    void                adjust_start(const T& new_start);

    void                adjust_end(const T& new_end);

    void                link_with(Interval<T>& other);

private:

    T       _a;
    T       _b;
    T       _h = 0;
    int     _left = 0;
    int     _right = 0;
    bool    _is_point = false;
    int     _dir = 0;
};



template<typename T, size_t N>
class Interpolator{

public:

    virtual ~Interpolator() = default;

    Interpolator() = delete;

    //ACCESSORS

    virtual const Interval<T>&          interval() const = 0;

    virtual const Array1D<T, N>&        q_start() const = 0;

    virtual const Array1D<T, N>&        q_end() const = 0;

    virtual int                         start_bdr() const = 0;

    virtual int                         end_bdr() const = 0;

    virtual int                         dir() const = 0;

    virtual const T&                    t_start() const = 0;

    virtual const T&                    t_end() const = 0;

    virtual bool                        is_out_of_bounds(const T& t) const = 0;

    virtual bool                        can_link_with(const Interpolator<T, N>& other) const = 0;

    virtual Interpolator<T, N>*         clone() const = 0;

    std::unique_ptr<Interpolator<T, N>> safe_clone() const;

    size_t                              array_size() const;

    void                                call(T* result, const T& t) const;

    Array1D<T, N>                       make_call(const T& t) const;


    //MODIFIERS

    virtual void link_with(Interpolator<T, N>& other) = 0;

    virtual void link_after(Interpolator<T, N>& other) = 0;

    virtual void adjust_start(const T& t_start) = 0;

    virtual void adjust_end(const T& t_end) = 0;

    virtual void close_end() = 0;

    virtual void close_start() = 0;

    virtual void open_end() = 0;

protected:

    Interpolator(size_t n);

    DEFAULT_RULE_OF_FOUR(Interpolator);

private:

    virtual void _call_impl(T* result, const T& t) const = 0;

    size_t _array_size;

};




template<typename T, size_t N>
class LocalInterpolator : public Interpolator<T, N>{
    //The purpose of this class is to provide a continuous output between the naturally adapted states
    //of an OdeSolver.
    
public:

    LocalInterpolator() = delete;

    LocalInterpolator(const T& t, const T* q, size_t size);

    LocalInterpolator(T t1, T t2, const T* y1, const T* y2, size_t size, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(LocalInterpolator);

    //ACCESSORS

     const Interval<T>&       interval() const override;

     const Array1D<T, N>&     q_start() const override;

     const Array1D<T, N>&     q_end() const override;

    virtual size_t                  order() const;

     int                      start_bdr() const override;

     int                      end_bdr() const override;

     int                      dir() const override;

     const T&                 t_start() const override;

     const T&                 t_end() const override;

     bool                     is_out_of_bounds(const T& t) const override;

     bool                     can_link_with(const Interpolator<T, N>& other) const override;

    LocalInterpolator<T, N>*        clone() const override;

    //MODIFIERS

    void                            adjust_start(const T& t_start) override;

    void                            adjust_end(const T& t_end) override;

    void                            link_with(Interpolator<T, N>& other) override;

    void                            link_after(Interpolator<T, N>& other) override;

     void                     close_end() override;

     void                     close_start() override;

     void                     open_end() override;

protected:

    const T& _t_min() const;

    const T& _t_max() const;

private:

    void _call_impl(T* result, const T& t) const override;

    Interval<T> _interval;
    T _tmin;
    T _tmax;
    Array1D<T, N> _q_old;
    Array1D<T, N> _q;

};





template<typename T, size_t N>
class StandardLocalInterpolator final: public LocalInterpolator<T, N>{
    
public:

    StandardLocalInterpolator() = delete;

    StandardLocalInterpolator(const T& t, const T* q, size_t size);

    StandardLocalInterpolator(const Array2D<T, N, 0>& coef_mat, T t1, T t2, const T* y1, const T* y2, size_t size, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(StandardLocalInterpolator);

    size_t order() const override;

    StandardLocalInterpolator<T, N>* clone() const override;

protected:

    Array2D<T, N, 0> _coef_mat; //number of columns is equal to the interpolation order
    size_t _order = 0;

private:

    void _call_impl(T* result, const T& t) const override;


};



template<typename T, size_t N, typename INTERPOLATOR=Interpolator<T, N>>
class LinkedInterpolator : public Interpolator<T, N>{

    static constexpr bool _is_void = std::is_same_v<INTERPOLATOR, Interpolator<T, N>>;

public:

    LinkedInterpolator() = delete;

    LinkedInterpolator(const INTERPOLATOR* other);

    LinkedInterpolator(const T& t, const T* q, size_t size);

    LinkedInterpolator(const LinkedInterpolator& other);

    LinkedInterpolator(LinkedInterpolator&& other) = default;

    LinkedInterpolator& operator=(const LinkedInterpolator& other);

    LinkedInterpolator& operator=(LinkedInterpolator&& other) = default;

    //ACCESSORS

    const Interval<T>&                      interval() const override;

    size_t                                  size() const;

    const Array1D<T, N>&                    q_start() const override;

    const Array1D<T, N>&                    q_end() const override;

    int                                     start_bdr() const override;

    int                                     end_bdr() const override;

     int                              dir() const override;

     const T&                         t_start() const override;

     const T&                         t_end() const override;

     const INTERPOLATOR&              interpolant(size_t i) const;

     const INTERPOLATOR&              last_interpolant() const;

    bool                                    is_out_of_bounds(const T& t) const override;

     bool                             can_link_with(const Interpolator<T, N>& interpolant) const override;

    LinkedInterpolator<T, N, INTERPOLATOR>* clone() const override;


    //MODIFIERS

    void                                    link_with(Interpolator<T, N>& interpolant) override;

    void                                    link_after(Interpolator<T, N>& interpolant) override;

     void                             adjust_start(const T& t_new) override;

     void                             adjust_end(const T& t_new) override;

    void                                    expand(const INTERPOLATOR& interpolant);

    void                                    expand_by_owning(std::unique_ptr<Interpolator<T, N>>&& interpolant) requires _is_void;

     void                             open_end() override;

     void                             close_end() override;

     void                             close_start() override;

private:

    size_t                              _search_index(const T& t) const;

     bool                         _can_replace_last_with(const Interpolator<T, N>& other) const;

    void                                _call_impl(T* result, const T& t) const override;

           INTERPOLATOR&          _get(size_t i);

     const INTERPOLATOR&          _get(size_t i) const;

           INTERPOLATOR&          _get_last();

     const INTERPOLATOR&          _get_last() const;
    
           INTERPOLATOR&          _get_safe(size_t i);

     const INTERPOLATOR&          _get_safe(size_t i) const;

     void                         _throw_invalid_interpolant(const Interpolator<T, N>& interpolant) const;

    std::vector<INTERPOLATOR> _interpolants = {};
    std::vector<std::unique_ptr<Interpolator<T, N>>> _interp_ptrs = {};
    int _dir = 0;
    mutable bool _interval_cached = false;
    mutable Interval<T> _interval;

};



template<typename T, size_t N=0>
class OdeSolution : public OdeResult<T, N>{

public:

    OdeSolution(const std::vector<T>& t, const Array2D<T, 0, N>& q, const EventMap& event_map, bool diverges, bool success, double runtime, std::string message, const Interpolator<T, N>& interpolator);

    OdeSolution(const OdeSolution& other);

    OdeSolution(OdeSolution&& other) noexcept;

    OdeSolution(OdeResult<T, N>&& other, const Interpolator<T, N>& interpolator);

    OdeSolution& operator=(const OdeSolution& other);

    OdeSolution& operator=(OdeSolution&& other) noexcept;

    ~OdeSolution();

    Array1D<T, N> operator()(const T& t) const;

    OdeSolution<T, N>* clone() const override;

private:

    Interpolator<T, N>* _interpolator;

};


template<typename T>
void lin_interp(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, size_t size);

template<typename T>
void coef_mat_interp(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size);



} // namespace ode

#endif