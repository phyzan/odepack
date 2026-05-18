#ifndef MCMC_HPP
#define MCMC_HPP


#include <functional>
#include <random>
#include <memory>
#include <chrono>
#include "tools.hpp"

namespace mcmc{

template<typename Scalar, typename State>
using Observable = std::function<Scalar(const State&)>;


template<typename Scalar>
class MarkovChain;

template<typename Scalar, typename State, typename Derived>
class DerivedMarkovChain;



template<typename Scalar>
class MonteCarlo;




template<typename Scalar>
class MarkovChain{

public:

    virtual ~MarkovChain() = default;

    virtual MarkovChain*                 clone() const = 0;

    virtual std::unique_ptr<MarkovChain> safe_clone() const = 0;

    virtual void                         update(const std::string& method, const size_t& steps=1) = 0;

protected:

    MarkovChain() = default;

    DEFAULT_RULE_OF_FOUR(MarkovChain)

};




template<typename Scalar, typename State, typename Derived>
class DerivedMarkovChain : public MarkovChain<Scalar>{

public:
    using propagator = void (Derived::*)();

    const State&                         derived_state() const;

    MarkovChain<Scalar>*                 clone() const final;

    std::unique_ptr<MarkovChain<Scalar>> safe_clone() const final;

    inline void                          update(const std::string& method, const size_t& steps=1) final;

    void                                 update(propagator method, const size_t& steps);

    inline Scalar                        draw_uniform(const Scalar& a, const Scalar& b) const;

    inline propagator                    method(const std::string& name) const; //virtual


protected:


    DerivedMarkovChain(const State& initial_state);

    DerivedMarkovChain();

    DEFAULT_RULE_OF_FOUR(DerivedMarkovChain)

    State _state;
    mutable std::mt19937 _gen;


private:

    void _init_gen();

    mutable std::uniform_real_distribution<Scalar> _uniform_dist;

};


template<typename Scalar>
class MonteCarlo{


public:

    virtual ~MonteCarlo() = default;

    DEFAULT_RULE_OF_FOUR(MonteCarlo)

    virtual size_t                      N() const = 0;

    virtual void                        update(const std::string& method, const size_t& steps, const size_t& sweeps=0) = 0;

    virtual void                        thermalize(const std::string& method, const size_t& sweeps) = 0;

    virtual const MarkovChain<Scalar>&  chain() const = 0;

    virtual MonteCarlo<Scalar>*         clone() const = 0;

    virtual std::unique_ptr<MonteCarlo<Scalar>> safe_clone() const = 0;

protected:

    MonteCarlo() = default;


};






template<typename Scalar, typename State, typename Chain, typename Derived>
class DerivedMonteCarlo : public MonteCarlo<Scalar>{

public:

    Sample<Scalar>                      sample(const Observable<Scalar, State>& A) const;

    const std::vector<State>&           data() const;

    void                                update(const std::string& method, const size_t& steps, const size_t& sweeps=0) final;

    void                                thermalize(const std::string& method, const size_t& sweeps) final;

    const MarkovChain<Scalar>&          chain() const final;

    size_t                              N() const final;

    MonteCarlo<Scalar>*                 clone() const final;

    std::unique_ptr<MonteCarlo<Scalar>> safe_clone() const;

protected:

    DerivedMonteCarlo(const Chain& mc) : MonteCarlo<Scalar>(), _mc(mc) {}

    DEFAULT_RULE_OF_FOUR(DerivedMonteCarlo)


    Chain _mc;
    std::vector<State> _data;
};


template<typename Scalar>
void update_all(const std::vector<MonteCarlo<Scalar>*>& obj, const std::string& method, const size_t& steps, const size_t& sweeps, int threads);



/*
---------------------------------------------------------------------------
---------------------------------------------------------------------------
------------------------------DEFINITIONS----------------------------------
---------------------------------------------------------------------------
---------------------------------------------------------------------------
*/


template<typename Scalar, typename State, typename Derived>
DerivedMarkovChain<Scalar, State, Derived>::DerivedMarkovChain(const State& initial_state): MarkovChain<Scalar>(), _state(initial_state), _gen(std::random_device()()),  _uniform_dist(0, 1){
    _init_gen();
}

template<typename Scalar, typename State, typename Derived>
DerivedMarkovChain<Scalar, State, Derived>::DerivedMarkovChain(): MarkovChain<Scalar>(), _gen(std::random_device()()),  _uniform_dist(0, 1){
    _init_gen();
}

template<typename Scalar, typename State, typename Derived>
inline Scalar DerivedMarkovChain<Scalar, State, Derived>::draw_uniform(const Scalar& a, const Scalar& b) const{
    return this->_uniform_dist(_gen)*(b-a) + a;
}


template<typename Scalar, typename State, typename Derived>
inline typename DerivedMarkovChain<Scalar, State, Derived>::propagator DerivedMarkovChain<Scalar, State, Derived>::method(const std::string& name) const{
    return static_cast<const Derived*>(this)->method(name);
}



template<typename Scalar, typename State, typename Derived>
const State& DerivedMarkovChain<Scalar, State, Derived>::derived_state() const{
    return _state;
}

template<typename Scalar, typename State, typename Derived>
MarkovChain<Scalar>* DerivedMarkovChain<Scalar, State, Derived>::clone() const {
    return new Derived(*static_cast<const Derived*>(this));
}

template<typename Scalar, typename State, typename Derived>
std::unique_ptr<MarkovChain<Scalar>> DerivedMarkovChain<Scalar, State, Derived>::safe_clone() const{
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
}


template<typename Scalar, typename State, typename Derived>
inline void DerivedMarkovChain<Scalar, State, Derived>::update(const std::string& method, const size_t& steps) {
    this->update(this->method(method), steps);
}



template<typename Scalar, typename State, typename Derived>
void DerivedMarkovChain<Scalar, State, Derived>::update(propagator method, const size_t& steps){
    
    for (size_t i=0; i<steps; i++){
        (static_cast<Derived*>(this)->*method)();
    }
}


template<typename Scalar, typename State, typename Derived>
void DerivedMarkovChain<Scalar, State, Derived>::_init_gen(){
    _gen.seed(static_cast<std::mt19937::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
}




template<typename Scalar, typename State, typename Chain, typename Derived>
void DerivedMonteCarlo<Scalar, State, Chain, Derived>::thermalize(const std::string& method, const size_t& sweeps) {
    this->_mc.update(method, sweeps);
}



template<typename Scalar, typename State, typename Chain, typename Derived>
const MarkovChain<Scalar>& DerivedMonteCarlo<Scalar, State, Chain, Derived>::chain() const{
    return _mc;
}


template<typename Scalar, typename State, typename Chain, typename Derived>
size_t DerivedMonteCarlo<Scalar, State, Chain, Derived>::N() const {
    return _data.size();
}

template<typename Scalar, typename State, typename Chain, typename Derived>
const std::vector<State>& DerivedMonteCarlo<Scalar, State, Chain, Derived>::data() const{
    return _data;
}


template<typename Scalar, typename State, typename Chain, typename Derived>
Sample<Scalar> DerivedMonteCarlo<Scalar, State, Chain, Derived>::sample(const Observable<Scalar, State>& A)const{
    std::vector<Scalar> sample_array(this->N());
    for (size_t i=0; i<N(); i++){
        sample_array[i] = A(_data[i]);
    }
    return sample_array;
}


template<typename Scalar, typename State, typename Chain, typename Derived>
MonteCarlo<Scalar>* DerivedMonteCarlo<Scalar, State, Chain, Derived>::clone() const {
    return new Derived(*static_cast<const Derived*>(this));
}

template<typename Scalar, typename State, typename Chain, typename Derived>
std::unique_ptr<MonteCarlo<Scalar>> DerivedMonteCarlo<Scalar, State, Chain, Derived>::safe_clone() const{
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
}


template<typename Scalar, typename State, typename Chain, typename Derived>
void DerivedMonteCarlo<Scalar, State, Chain, Derived>::update(const std::string& method, const size_t& steps, const size_t& sweeps) {
    typename Chain::propagator pr = this->_mc.method(method);
    for (size_t i=0; i<steps; i++){
        this->_mc.update(pr, sweeps+1);
        this->_data.push_back(this->_mc.derived_state());
    }
}


template<typename Scalar>
void update_all(const std::vector<MonteCarlo<Scalar>*>& obj, const std::string& method, const size_t& steps, const size_t& sweeps, int threads){

    threads = (threads <= 0) ? omp_get_max_threads() : threads;
    #pragma omp parallel for num_threads(threads)
    for (size_t i=0; i<obj.size(); i++){
        obj[i]->update(method, steps, sweeps);
    }
}

} // namespace mcmc

#endif // MCMC_HPP