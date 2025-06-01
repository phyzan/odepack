#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <map>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <fstream>


// USEFUL ALIASES

template<class T, int N=-1>
using vec = Eigen::Array<T, N, 1>;

template<class T, int N, template<class, int> typename VecLike=vec>
using Func = std::function<VecLike<T, N>(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N, template<class, int> typename VecLike=vec>
using Fvoid = std::function<void(VecLike<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N, template<class, int> typename VecLike=vec>
using Fptr = VecLike<T, N>(*)(const T&, const vec<T, N>&, const std::vector<T>&);

template<class T, int N, template<class, int> typename VecLike=vec>
using Fvoidptr = void(*)(VecLike<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&);

template<class T>
using _ObjFun = std::function<T(const T&)>;

using _Shape = std::vector<size_t>;

template<class T>
using complex = std::complex<T>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag;

template <typename T>
constexpr T inf() {
    return std::numeric_limits<T>::infinity();
}

template<class T, int N>
T norm_squared(const vec<T, N>& f){
    return (f*f).sum();
}

template<class T, int N>
T rms_norm(const vec<T, N>& f){
    return sqrt(norm_squared(f) / f.size());
}

template<class T>
T abs(const T& x){
    return (x > 0) ? x : -x;
}


template<class T, int N>
void write_checkpoint(std::ofstream& file, const T& t, const vec<T, N>& q, const int& event_index);



template<class T>
std::vector<T> subvec(const std::vector<T>& x, const size_t& start) {
    if (start >= x.size()) {
        return {}; // Return an empty vector if start is out of bounds
    }
    return std::vector<T>(x.begin() + start, x.end());
}


template<class T>
std::vector<T> _event_data(const std::vector<T>& q, const std::map<std::string, std::vector<size_t>>& event_map, const std::string& event);

//BISECTION USED FOR EVENTS IN ODES

template<class T>
std::vector<T> bisect(const _ObjFun<T>& f, const T& a, const T& b, const T& atol){
    T err = 2*atol+1;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    if (f(a)*f(b) > 0){
        throw std::runtime_error("Root not bracketed");
    }

    while (err > atol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            break;
        }
        fm = f(c);
        if (f(_a) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = abs(fm);
    }
    return {_a, c, _b};
}


template<class T>
void mat_vec_prod(T* result, const T* mat, const T* vec, const size_t& rows, const size_t& cols, const T& factor=1){
    /*
    result[i] = sum_j mat[i, j] * vec[j]
    */
    for (size_t i=0; i<rows; i++){
        T _sum = 0;
        for (size_t j=0; j<cols; j++){
            _sum += mat[i*cols+j]*vec[j];
        }
        result[i] = _sum*factor;
    }
}

template<class T>
void mat_T_vec_prod(T* result, const T* mat, const T* vec, const size_t& rows, const size_t& cols, const T& factor=1){
    /*
    The same as above but the transpose matrix is used
    */
    for (size_t i=0; i<cols; i++){
        T _sum = 0;
        for (size_t j=0; j<rows; j++){
            _sum += mat[j*cols+i]*vec[j];
        }
        result[i] = _sum*factor;
    }
}

template<class S>
void mat_mat_prod(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n, const S& factor=1){
    /*
    a : (m x s)
    b : (s x n)
    */
    for (size_t k=0; k<m*n; k++){
        size_t i = k/n;
        size_t j = k % n;
        S _sum = 0;
        for (size_t q=0; q<s; q++){
            _sum += a[i*s + q] * b[q*n + j];
        }
        r[i*n+j] = _sum*factor;
    }
}


template<class S>
void mat_T_mat_prod(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n, const S& factor=1){
    /*
    a : (s x m)
    b : (s x n)
    */
    for (size_t k=0; k<m*n; k++){
        size_t i = k/n;
        size_t j = k % n;
        S _sum = 0;
        for (size_t q=0; q<s; q++){
            _sum += a[q*n+j] * b[q*n + j];
        }
        r[i*n+j] = _sum*factor;
    }
}


template<class T, int N, template<class, int> typename VecLike=vec>
struct Functor{

    Functor(){}

    Functor(std::nullptr_t ptr) : func(nullptr){}

    Functor(const Fvoid<T, N, VecLike>& f):func(f){}

    Functor(const Func<T, N, VecLike>& f): func([f](VecLike<T, N>& res, const T& t, const VecLike<T, N>& q, const std::vector<T>& args){res = f(t, q, args); }){}

    Functor(const Fvoidptr<T, N, VecLike>& f):func(f){}

    Functor(const Fptr<T, N, VecLike>& f): Functor([f](VecLike<T, N>& res, const T& t, const VecLike<T, N>& q, const std::vector<T>& args){res = f(t, q, args); }){}

    inline void operator()(VecLike<T, N>& result, const T& t, const VecLike<T, N>& q, const std::vector<T>& args)const{
        func(result, t, q, args);
    }

    inline VecLike<T, N> operator()(const T& t, const VecLike<T, N>& q, const std::vector<T>& args)const{
        VecLike<T, N> res(q.size());
        func(res, t, q, args);
        return res;
    }

    Functor<T, N, VecLike>& operator=(const Fvoid<T, N, VecLike>& new_func){
        func = new_func;
        return *this;
    }

    Functor<T, N, VecLike>& operator=(const Func<T, N, VecLike>& f){
        func = [f](VecLike<T, N>& res, const T& t, const VecLike<T, N>& q, const std::vector<T>& args){res = f(t, q, args); };
        return *this;
    }

    bool operator==(const Functor<T, N, VecLike>& other){
        return (this == &other) ? true : other.func == func;
    }

    template<class Any>
    bool operator==(const Any& other){
        return func == other;
    }

    Fvoid<T, N, VecLike> func=nullptr;
};


//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<class T, int N>
struct OdeResult{


    const std::vector<T> t;
    const std::vector<vec<T, N>> q;
    const std::map<std::string, std::vector<size_t>> event_map;
    const bool diverges;
    const bool success;// if the OdeSolver didnt die during the integration
    const double runtime;
    const std::string message;

    void examine() const{
        std::cout << std::endl << "OdeResult\n------------------------\n------------------------\n" <<
        "\tPoints           : " << t.size() << "\n" <<
        "\tDiverges         : " << (diverges ? "true" : "false") << "\n" << 
        "\tSuccess          : " << (success ? "true" : "false") << "\n" <<
        "\tRuntime          : " << runtime << "\n" <<
        "\tTermination cause: " << message << "\n" <<
        event_log();
    }

    std::string event_log() const{
        std::string res = "";
        res += "\tEvents:\n\t----------\n";
        for (const auto& [name, array] : event_map){
            res += "\t    " + name + " : " + std::to_string(array.size()) + "\n";
        }
        res += "\n\t----------\n";
        return res;
    }

    std::vector<T> t_filtered(const std::string& event) const {
        return _event_data(this->t, this->event_map, event);
    }

    std::vector<vec<T, N>> q_filtered(const std::string& event) const {
        return _event_data(this->q, this->event_map, event);
    }
    
};


template<class T, int N>
class SolverState{

public:
    const T t;
    const vec<T, N> q;
    const T habs;
    const std::string event;
    const bool diverges;
    const bool is_running; //if tmax or breakcond are met or is dead, it is set to false. It can be set to true if new tmax goal is set
    const bool is_dead; //This is irreversible.
    const size_t Nt;
    const std::string message;

    SolverState(const T& t, const vec<T, N>& q, const T& habs, const std::string& event, const bool& diverges, const bool& is_running, const bool& is_dead, const size_t& Nt, const std::string& message): t(t), q(q), habs(habs), event(event), diverges(diverges), is_running(is_running), is_dead(is_dead), Nt(Nt), message(message) {}

    void show(const int& precision = 15) const{

        std::cout << std::endl << std::setprecision(precision) << 
        "OdeSolver current state:\n---------------------------\n"
        "\tt          : " << t << "\n" <<
        "\tq          : " << q << "\n" <<
        "\th          : " << habs << "\n\n";
        std::cout << ((event == "") ? "\tNo event" : "\tEvent      : " + (event) )<< "\n" <<
        "\tDiverges   : " << (diverges ? "true" : "false") << "\n" << 
        "\tRunning    : " << (is_running ? "true" : "false") << "\n" <<
        "\tUpdates    : " << Nt << "\n" <<
        "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
        "\tState      : " << message << "\n";
    }




};


template<class T, int N>
struct State{

    T t; //current time
    vec<T, N> q; //current vector
    T habs; //absolute stepsize to be used for the next step
    int direction = 0;

    virtual ~State(){}

    State(const T& t, const vec<T, N>& q, const T& habs):t(t), q(q), habs(habs){}

    State() = delete;

    virtual State<T, N>* clone() const{
        return new State<T, N>(*this);
    }

    virtual State<T, N>& assign(const State<T, N>& other){
        return this->operator=(other);
    }

    inline void set_direction(const T& dir){
        direction = (dir == 0) ? 0 : ( (dir > 0) ? 1 : -1);
    }
    
    virtual void adjust(const T& h_abs, const T& dir, const vec<T, N>& diff){
        habs = h_abs;
        set_direction(dir);
    }

    T h() const{
        return habs*direction;
    }

protected:

    State(const State<T, N>& other):t(other.t), q(other.q), habs(other.habs){}

    State<T, N>& operator=(const State<T, N>& other) = default;
};


template<class T>
std::vector<T> _event_data(const std::vector<T>& q, const std::map<std::string, std::vector<size_t>>& event_map, const std::string& event){
    std::vector<size_t> ind = event_map.at(event);
    std::vector<T> data(ind.size());
    for (size_t i=0; i<data.size(); i++){
        data[i] = q[ind[i]];
    }
    return data;
}


template<class T, int N>
void write_checkpoint(std::ofstream& file, const T& t, const vec<T, N>& q, const int& event_index){
    file << event_index << " " << std::setprecision(16) << t;
    for (size_t i=0; i<static_cast<size_t>(q.size()); i++){
        file << " " << std::setprecision(16) << q[i];
    }
    file << "\n";
}


template<class Any, class T, int N>
bool operator==(const Any& a, const Functor<T, N>& b){
    return b.func == a;
}


template<class T, int N>
struct _MutableData{

    _MutableData(const State<T, N>& state) : q(state.q), qdiff(state.q), state(state.clone()){}

    _MutableData(const _MutableData<T, N>& other): q(other.q), qdiff(other.qdiff){
        state = other.state.clone();
    }

    _MutableData(_MutableData<T, N>&& other):q(std::move(other.q)), qdiff(std::move(other.qdiff)), state(other.state){
        other.state = nullptr;
    }

    _MutableData(){}


    ~_MutableData(){
        delete state;
        state = nullptr;
    }

    _MutableData<T, N>& operator=(const _MutableData<T, N>& other){
        if (&other == this) return *this;
        q = other.q;
        qdiff = other.qdiff;
        delete state;
        state = nullptr;
        if (other.state != nullptr){
            state = other.state->clone();
        }
        return *this;
    }

    vec<T, N> q;
    vec<T, N> qdiff;
    State<T, N>* state=nullptr;
};



template<class T, int N>
class Checkpoint{

    //once the checkpoint has been initialized with a state,
    //then the same type of state be reassigned, otherwise object slicing might occur.
    //However if a different type of state is needed, then a new checkpoint
    //has to be created (which can safely be assigned to the original without internal object slicing.)

public:
    Checkpoint(const State<T, N>& state): _state(state.clone()){
        _is_set = true;
    }

    Checkpoint(const Checkpoint<T, N>& other){
        _copy_checkpoint(other);
    }

    Checkpoint(Checkpoint<T, N>&& other) : _state(other._state){
        other._state = nullptr;
    }

    Checkpoint(){}

    ~Checkpoint(){
        delete _state;
    }

    Checkpoint<T, N>& operator=(const Checkpoint<T, N>& other){
        if (&other == this) return *this;
        _copy_checkpoint(other);
        return *this;
    }

    const State<T, N>& state()const{

        if (!is_set()){
            throw std::runtime_error("Checkpoint has not been set.");
        }
        return *_state;
    }

    inline bool is_set()const{
        return _is_set;
    }

    void set(const State<T, N>& state){
        if (_state == nullptr){
            _state = state.clone();
        }
        else{
            _state->assign(state);
        }
        _is_set = true;
    }

    inline void remove(){
        _is_set = false;
    }



private:

    State<T, N>* _state = nullptr;
    bool _is_set = false;

    void _copy_checkpoint(const Checkpoint<T, N>& other){
        delete _state;
        _state = nullptr;
        if (other._state != nullptr){
            _state = other._state->clone();
        }

        _is_set = other._is_set;
    }

};


#endif