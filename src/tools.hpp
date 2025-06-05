#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <iostream>
#include <map>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <fstream>
#include <chrono>

// USEFUL ALIASES

template<class T, int N=-1>
using vec = Eigen::Array<T, N, 1>;

template<class T, int N, template<class, int> typename VecLike=vec>
struct Functor;

template<class T, int N, template<class, int> typename VecLike=vec>
using Func = std::function<VecLike<T, N>(const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N, template<class, int> typename VecLike=vec>
using Fvoid = std::function<void(VecLike<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&)>;

template<class T, int N, template<class, int> typename VecLike=vec>
using Fptr = VecLike<T, N>(*)(const T&, const vec<T, N>&, const std::vector<T>&);

template<class T, int N, template<class, int> typename VecLike=vec>
using Fvoidptr = void(*)(VecLike<T, N>&, const T&, const vec<T, N>&, const std::vector<T>&);

template<class T, int N>
using JacMat = Eigen::Matrix<T, N, N>;

template<class T, int N>
using Jac = Functor<T, N, JacMat>;

template<class T>
using _ObjFun = std::function<T(const T&)>;

using _Shape = std::vector<size_t>;

template<class T>
using complex = std::complex<T>;

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag;

using TimePoint = std::chrono::_V2::system_clock::time_point;

TimePoint now(){
    return std::chrono::high_resolution_clock::now();
}

double timeit(const TimePoint& t1, const TimePoint& t2){
    std::chrono::duration<double> duration = t2-t1;
    return duration.count();
}

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


std::string format_duration(const double& t){
    int h = t/3600;
    int m = (t - h*3600)/60;
    int s = (t - h*3600 - m*60);

    return std::to_string(h) + " h, " + std::to_string(m) + " m, " + std::to_string(s) + " s";  
}


class Clock{

public:

    Clock(){}

    void start(){
        _start = now();
    }

    double seconds() const{
        return timeit(_start, now());
    }

    std::string message() const{
        return format_duration(seconds());     
    }

private:

    TimePoint _start;
};


void show_progress(const int& n, const int& target, const Clock& clock){
    std::cout << "\033[2K\rProgress: " << std::setprecision(2) << n*100./target << "%" <<   " : " << n << "/" << target << "  Time elapsed : " << clock.message() << "      Estimated duration: " << format_duration(target*clock.seconds()/n) << std::flush;
}


template<class T, int N, template<class, int> typename VecLike>
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


template<class T, int N>
struct OdeRhs{

    Functor<T, N> ode_rhs=nullptr;
    Jac<T, N> jacobian=nullptr;

    template<class A>
    OdeRhs(const A& rhs) : ode_rhs(rhs), jacobian(nullptr){}

    template<class A, class B>
    OdeRhs(const A& rhs, const B& jac) : ode_rhs(rhs), jacobian(jac){}
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
        "\tq          : " << q.transpose() << "\n" <<
        "\th          : " << habs << "\n\n";
        std::cout << ((event == "") ? "\tNo event" : "\tEvent      : " + (event) )<< "\n" <<
        "\tDiverges   : " << (diverges ? "true" : "false") << "\n" << 
        "\tRunning    : " << (is_running ? "true" : "false") << "\n" <<
        "\tUpdates    : " << Nt << "\n" <<
        "\tDead       : " << (is_dead ? "true" : "false") << "\n" <<
        "\tState      : " << message << "\n";
    }




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


template<class T, int N, class StateDerived>
struct _MutableData{

    _MutableData(const StateDerived& state) : q(state.vector()), qdiff(state.vector()), state(static_cast<StateDerived*>(state.clone())){}

    _MutableData(const _MutableData<T, N, StateDerived>& other): q(other.q), qdiff(other.qdiff){
        state = static_cast<StateDerived*>(other.state->clone());
    }

    _MutableData(_MutableData<T, N, StateDerived>&& other):q(std::move(other.q)), qdiff(std::move(other.qdiff)), state(other.state){
        other.state = nullptr;
    }

    _MutableData(){}


    ~_MutableData(){
        delete state;
        state = nullptr;
    }

    _MutableData<T, N, StateDerived>& operator=(const _MutableData<T, N, StateDerived>& other){
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
    StateDerived* state=nullptr;
};



template<class STATE>
class Checkpoint{

    //once the checkpoint has been initialized with a state,
    //then the same type of state be reassigned, otherwise object slicing might occur.
    //However if a different type of state is needed, then a new checkpoint
    //has to be created (which can safely be assigned to the original without internal object slicing.)

public:
    Checkpoint(const STATE& state): _state(state.clone()){
        _is_set = true;
    }

    Checkpoint(const Checkpoint<STATE>& other){
        _copy_checkpoint(other);
    }

    Checkpoint(Checkpoint<STATE>&& other) : _state(other._state){
        other._state = nullptr;
    }

    Checkpoint(){}

    ~Checkpoint(){
        delete _state;
    }

    Checkpoint<STATE>& operator=(const Checkpoint<STATE>& other){
        if (&other == this) return *this;
        _copy_checkpoint(other);
        return *this;
    }

    const STATE& state()const{

        if (!is_set()){
            throw std::runtime_error("Checkpoint has not been set.");
        }
        return *_state;
    }

    inline bool is_set()const{
        return _is_set;
    }

    void set(const STATE& state){
        if (_state == nullptr){
            _state = new STATE(state);
        }
        else{
            *_state = state;
        }
        _is_set = true;
    }
    
    inline void remove(){
        _is_set = false;
    }



private:

    STATE* _state = nullptr;
    bool _is_set = false;

    void _copy_checkpoint(const Checkpoint<STATE>& other){
        delete _state;
        _state = nullptr;
        if (other._state != nullptr){
            _state = new STATE(*other._state);
        }

        _is_set = other._is_set;
    }

};


#endif