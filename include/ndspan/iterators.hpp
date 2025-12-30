#ifndef ITERATORS_HPP
#define ITERATORS_HPP

#include "ndtools.hpp"

template<typename Derived, size_t ND>
class IndexIterator{

    static_assert(ND>0, "ND>0 in Index Iterator");
public:

    template<INT_T... IntType>
    INLINE bool constexpr iterating(IntType&... idx) const{
        static_assert(sizeof...(idx)==ND, "Invalid number of indices");
        return iterating_impl(idx...);
    }

    template<typename Callable>
    INLINE void constexpr iterate(Callable&& f) const{
        //static override
        return THIS_C->iterate(f);
    }

    template<INT_T... IntType>
    INLINE bool iterating_impl(IntType&... idx) const{
        //static override
        return THIS_C->iterating_impl(idx...);
    }

    inline size_t ndims() const{
        return ND;
    }

protected:

    IndexIterator() = default;

};

template<typename Derived, size_t ND>
class BaseNdIterator : public IndexIterator<Derived, ND>{


public:

    using IdxHolder = std::array<size_t, ND>;

    template<size_t I, typename Callable>
    INLINE static void constexpr iterate_impl(IdxHolder& idx, Callable&& f, const IdxHolder& limit){
        if constexpr (I < ND) {
            idx[I] = 0;
            do {
                iterate_impl<I+1>(idx, f, limit);
                idx[I]++;
            }while (idx[I] < limit[I]);
        }
        else{
            EXPAND(size_t, ND, J, 
                f(idx[J]...);
            );
        }
    }

protected:

    template<size_t I, size_t... Is, INT_T... IntType>
    INLINE static constexpr bool increment(std::index_sequence<Is...>, const IdxHolder& limit, IntType&... idx){
        if constexpr (I==0) {
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, false)))||...);
        }
        else{
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, increment<I-1>(std::index_sequence<Is...>{}, limit, idx...))))||...);
        }
    }


};



template<size_t... Dim>
class StaticNDIterator : public BaseNdIterator<StaticNDIterator<Dim...>, sizeof...(Dim)>{

    static constexpr size_t ND = sizeof...(Dim);
    using Base = BaseNdIterator<StaticNDIterator<Dim...>, ND>;
    

    static constexpr typename Base::IdxHolder SHAPE = {Dim...};

public:

    template<INT_T... IntType>
    INLINE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, SHAPE, idx...);
    }

    template<typename Callable>
    INLINE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, f, SHAPE);
    }

};


template<size_t ND>
class DynamicNDIterator : public BaseNdIterator<DynamicNDIterator<ND>, ND>{

    using Base = BaseNdIterator<DynamicNDIterator<ND>, ND>;

public:

    template<INT_T... IntType>
    constexpr DynamicNDIterator(IntType... shape) : _shape{shape...}{
        static_assert(sizeof...(shape)==ND, "Invalid shape size");
    }

    template<INT_T... IntType>
    INLINE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, _shape, idx...);
    }

    template<typename Callable>
    INLINE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, f, _shape);
    }

    typename Base::IdxHolder _shape;

};


template<size_t Slots, size_t Rank, bool CountSlots>
class MultiSetIterator{

public:

    using SetType = std::array<size_t, Slots>;
    using CounterType = std::array<size_t, Rank>;

    template<typename Callable>
    INLINE static constexpr void apply_iter_on(Callable&& f){
        SetType set{};
        CounterType counter{};
        doit<0>(set, counter,f);
    }

    template<size_t slot, typename Callable>
    INLINE static void constexpr doit(SetType& set, CounterType& counter, Callable&& f){
        if constexpr (slot < Slots) {
            size_t& var = set[slot];
            if constexpr (slot == 0) {
                var = 0;
            }
            else{
                var = set[slot-1];
            }
            if constexpr (CountSlots) {
                counter[var]++;
            }

            do {
                doit<slot+1>(set, counter,f);
                if constexpr (CountSlots) {
                    counter[var++]--;
                    if (var < Rank){
                        counter[var]++;
                    }
                }
                else{
                    var++;
                }
            }while (var < Rank);
        }
        else{
            f(static_cast<const SetType&>(set), static_cast<const CounterType&>(counter));
        }
    }

};

inline constexpr size_t multiset_coef(size_t n, size_t k){
    //number of weak compositions
    //let A be a vector of n positive integers. 
    //This function returns the number of vectors
    // (combinations of (A_1, ... ,A_n)) that sum exactly to k
    return comb(n+k-1, k);
}



#endif