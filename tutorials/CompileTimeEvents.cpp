#include <odepack/odepack.hpp>

using namespace ode;


/*
Assuming that the solution of a 2D ODE system is accessed via q[0] and q[1], we define two objective functions that detect when q[0] crosses 1 and when q[1] crosses 2.
*/


// ====================== Struct declarations =========================

// Define the objective functions as structs with operator() overloads to allow
// the compiler to inline the calls by passing the structs as template parameters to the solver class.
template<typename T>
struct Obj1{
    inline T operator()(T t, const T* q) const{
        return q[0] - 1;
    }
};

template<typename T>
struct Obj2{
    inline T operator()(const T& t, const T* q) const{
        return q[1] - 2;
    }
};

// Define the ODE system, e.g. simple harmonic oscillator
struct MyOdeRhs{

    // Declare parameters as `auto` to allow for automatic differentiation with dual numbers
    inline static void Rhs(auto* out, const auto& t, const auto* q){
        out[0] = q[1];
        out[1] = -q[0];
    }
};


// ===================== Class declaration =========================

// Class that hardcodes the ODE system and the **two** objective functions,
// and uses the ObjectiveSolver to handle events.
template<SolverTemplate typename Solver, typename T, size_t N, SolverPolicy SP = SolverPolicy::Static>
class MySolver : public ObjectiveSolver<Solver, T, N, SP, MyOdeRhs, Obj1<T>, Obj2<T>>{

public:

    using Base = ObjectiveSolver<Solver, T, N, SP, MyOdeRhs, Obj1<T>, Obj2<T>>;

    // Following the convention that the first argument of solvers is the ODE system, we pass MyOdeRhs as the first argument to the base class constructor.
    // The rest of the arguments are forwarded to the base class constructor.
    template<typename... Args>
    MySolver(
        // ------------- Main constructor -------------
        std::tuple<T, int> data1, // {accuracy, direction} for Obj1
        std::tuple<T, int> data2, // {accuracy, direction} for Obj2
        Args&&... args) // arguments to be forwardes into `Solver`
        // ---------------------------------------------
        : Base(
            std::tuple{ // forward {objective_function, accuracy, direction} for each objective
                ObjFunData{Obj1<T>{}, std::get<0>(data1), std::get<1>(data1)},
                ObjFunData{Obj2<T>{}, std::get<0>(data2), std::get<1>(data2)}
            },
            MyOdeRhs{}, // ODE to be forwarded into `Solver`
            std::forward<Args>(args)... // the rest of the arguments to be forwarded into `Solver`
        ) {}
};


// ==================== Main function =========================
int main(){

    using T = double;

    const size_t count = 4;
    // Now advance each solver until `count` encounters with any objective functions have been detected.

    std::array<T, 2> q0 = {T{0.0}, T{5.0}}; // initial conditions


    // Using the `MySolver` class
    {
        std::cout << "---------- Using MySolver class ----------" << std::endl;
        auto solver = MySolver<RK45, T, 2>{
            {T{0.0}, 1},
            {T{0.0}, 1},
            T{0.0}, // initial time
            View1D<T, 2>{q0.data()}, // initial state
            T{1e-6}, // relative tolerance
            T{1e-6}, // absolute tolerance
            T{1e-6}, // minimum step size
            T{0.1}, // maximum step size
            T{0.01}, // initial step size
            1 // integration direction
        };

        size_t event_count = 0;
        while (event_count < count && solver.advance()){
            if (solver.is_at_objective()){
                std::cout << "Event detected at t = " << solver.t() << ", objective index = " << solver.current_objective() << std::endl;
                event_count++;
            }
        }
    }
    

    // Using ode::getObjectiveSolver by passing the same function as lambdas
    {
        std::cout << "\n---------- Using getObjectiveSolver with lambdas ----------" << std::endl;

        auto solver = getObjectiveSolver<RK45, T, 2>(
            std::tuple{
                ObjFunData{
                    // passing the lambda and ftol automatically deduces the template
                    // parameters for ObjFunData, so we don't need to explicitly specify them.
                    .func=[](T t, const T* q){
                        return q[0] - 1;
                    },
                    .ftol=T{0.0},
                    .dir=1
                },
                ObjFunData{
                    .func=[](T t, const T* q){
                        return q[0] - 2;
                    },
                    .ftol=T{0.0},
                    .dir=1
                },
            },
            OdeData{
                    // passing .Rhs as a lambda and .Jac as nullptr automatically deduces the template parameters for OdeData, so we don't need to explicitly specify them.
                    .Rhs=ODE_LAMBDA(out, t, q){
                        out[0] = q[1];
                        out[1] = -q[0];
                    },
                    .Jac=nullptr // for clarity (default is nullptr anyway)
                },
            T{0.0}, // initial time
            View1D<T, 2>{q0.data()}, // initial state
            T{1e-6}, // relative tolerance
            T{1e-6}, // absolute tolerance
            T{1e-6}, // minimum step size
            T{0.1}, // maximum step size
            T{0.01}, // initial step size
            1 // integration direction
        );

        size_t event_count = 0;
        while (event_count < count && solver.advance()){
            if (solver.is_at_objective()){
                std::cout << "Event detected at t = " << solver.t() << ", objective index = " << solver.current_objective() << std::endl;
                event_count++;
            }
        }
    }

    // For single events, use the `SingleObjectiveSolver` class:

    {
        std::cout << "\n---------- Using SingleObjectiveSolver class ----------" << std::endl;

        SingleObjectiveSolver<RK45, T, 2, ode::SolverPolicy::Static, MyOdeRhs, Obj1<T>> solver(
            ObjFunData{
                .func=Obj1<T>{},
                .ftol=T{0.0},
                .dir=1
            },
        MyOdeRhs{},
            T{0.0}, // initial time
            View1D<T, 2>{q0.data()}, // initial state
            T{1e-6}, // relative tolerance
            T{1e-6}, // absolute tolerance
            T{1e-6}, // minimum step size
            T{0.1}, // maximum step size
            T{0.01}, // initial step size
            1 // integration direction
        );

        size_t event_count = 0;
        while (event_count < count && solver.advance()){
            if (solver.is_at_objective()){
                std::cout << "Event detected at t = " << solver.t() << ", objective index = " << solver.current_objective() << std::endl;
                event_count++;
            }
        }
    }

    // or if direction does not matter, and maximum accuracy is desired, you can use the following constructor:
    {
        std::cout << "\n---------- Using SingleObjectiveSolver class with simpler constructor ----------" << std::endl;
        SingleObjectiveSolver<RK45, T, 2, ode::SolverPolicy::Static, MyOdeRhs, Obj1<T>> solver(
            Obj1<T>{},
            MyOdeRhs{},
            T{0.0}, // initial time
            View1D<T, 2>{q0.data()}, // initial state
            T{1e-6}, // relative tolerance
            T{1e-6}, // absolute tolerance
            T{1e-6}, // minimum step size
            T{0.1}, // maximum step size
            T{0.01}, // initial step size
            1 // integration direction
        );

        size_t event_count = 0;
        while (event_count < count && solver.advance()){
            if (solver.is_at_objective()){
                std::cout << "Event detected at t = " << solver.t() << ", objective index = " << solver.current_objective() << std::endl;
                event_count++;
            }
        }
    }

}

/*
g++ -std=c++20 -O3 -Iinclude -Iexternal/xdiff/include -Iexternal/xdiff/external/lazy/include -Iexternal/polybox/include -Iexternal/ndspan/include tutorials/CompileTimeEvents.cpp -o compile_time_events && ./compile_time_events
*/