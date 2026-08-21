#include "xdiff/tools.hpp"
#include <odepack/odepack.hpp>


using namespace ode;

int main(){

    using T = double;

    // Most common case: PreciseEvent class (zero - crossing event):

    /**
     * @class
        template<typename T, isObjFun<T> Target, OptionalRhsFunc<T> MaskFunc = std::nullptr_t>
        class PreciseEvent;
     * 
     * @tparam T : Numeric type
     * @tparam Target : satisfies isObjFun<T> concept:
        func(T t, const T* q) -> T
        when `func` crosses zero, the event is triggered
     * @tparam MaskFunc 
        Optionally modify the solver's state when the event is detected.
        Signature:
            mask(T* out, T t, const T* q) -> void
            or `nullptr` to *not* use the feature
     */

    // --------------- Directly declare events -----------------
    
    
    // Explicitly pass the `event_tol` as the required type (e.g. T{1e-15})
    // so that all template parameters are automatically deduced
    // and avoid writing PreciseEvent<...>
    PreciseEvent event{
        "my_event", // name
        [](const T& t, const T* q){
            return q[1] - 1; // Event triggered when q[1] = 1
        },
        T{1e-15}, // or 0.0 for maximum precision
        -1 // only trigger when the `Target` crosses from positive to negative
    };

    std::cout << "Created PreciseEvent `" << event.name() << "`\n" << std::endl;

    // ---------------- Polymorphic nature of Events -------------------

    // We want to allocate all events as Event<T>* for type-erasure
    Event<T>* raw_event = new PreciseEvent{
        "raw_event_pointer", // name
        [](const T& t, const T* q){
            return q[1] - 1; // Event triggered when q[1] = 1
        },
        T{1e-15}, // or 0.0 for maximum precision
        -1 // only trigger when the `Target` crosses from positive to negative
    };

    std::cout << "Successfully allocated `" << raw_event->name() << "`\n" << std::endl;

    delete raw_event;


    // ------------------ Safely allocate events ----------------------

    // The safe and provided way to do this is by allocating an Event
    // in a wrapper that holds unique ownership of the allocated Event<T>*
    // and works similarly to std::unique_ptr (cannot be copied, only moved)
    // See external/polybox/README.md for more details

    pbox::Box<Event<T>> event_1 = make_event<T, PreciseEvent>(
        "boxed_event_1",
        [](const T& t, const T* q){
            return q[1] - 1; // Event triggered when q[1] = 1
        },
        T{1e-15}, // or 0.0 for maximum precision
        -1 // only trigger when the `Target` crosses from positive to negative
    );

    pbox::Box<Event<T>> event_2 = make_event<T, PreciseEvent>(
        "boxed_event_2",
        [](const T& t, const T* q){
            return q[0] - 1.25; // Event triggered when q[0] = 1.25
        },
        T{0.0}
        // we leave `dir` default to 0 (any crossing direction)
    );

    std::cout << "Boxed event `" << event_1->name() << "` allocated!" << std::endl;
    std::cout << "Boxed event `" << event_2->name() << "` allocated!" << std::endl;
    // Now the allocated events are automatically destroyed when
    // `event_1` and `event_2` go out of scope. To transfer ownership,
    // call std::move, e.g.:

    Vector<pbox::Box<Event<T>>> event_list = xdiff::make_vector<pbox::Box<Event<T>>>(
        std::move(event_1),
        std::move(event_2)
    );

    // Now event_list is essentially an ode::Vector of event pointers
    // and has taken ownership of `event_1` and `event_2`, casting them invalid:

    if ((bool)event_1 || (bool)event_2){
        // Should not happen
        std::cout << "BAD: `event_1` or `event_2` still hold ownership of an event" << std::endl;
    } else {
        std::cout << "Success! Both `event_1` and `event_2` no longer have ownership of an event and only hold a nullptr" << std::endl;
    }

    // ----------------- using ode::make_event_list -------------------

    // Let's take back ownership in `event_1` and `event_2`

    event_1 = std::move(event_list[0]);
    event_2 = std::move(event_list[1]);

    // Now we can declare a solver with these events in a simple manner:

    std::array<T, 2> y0{3, 0};
    T t{0};
    T rtol{1e-10};
    T atol{1e-10};
    T min_step{0.0};
    T max_step{100};
    T stepsize{0};
    int dir = 1;

    auto solver = getSolver<RK45, ode::SolverPolicy::RichStatic>(
        OdeData{.Rhs=
            ODE_LAMBDA(out, /*t*/, q) {
                const T omega{1}; // angular frequency
                out[0] = q[1];
                out[1] = - omega*omega*q[0];
            }
        },
        t,
        View1D<T, 2>{y0.data()}, // q0
        rtol,
        atol,
        min_step,
        max_step,
        stepsize,
        dir,
        make_event_list<T>(std::move(event_1), std::move(event_2))
    );

    while (solver.advance() && solver.t() < 20){
        if (solver.at_event()){
            std::cout << "\n----------- EVENT DETECTED -----------" << std::endl;
            std::cout << "Event detected at t = " << solver.t() << ", event name = " << solver.current_event().event->name() << std::endl;
            std::cout << "State at event: ";
            auto v = solver.vector();
            for (size_t i = 0; i < 2; ++i) {
                std::cout << v[i] << " ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    // Print statements for `boxed_event_2` should be
    // twice as frequent as `boxed_event_1` since the latter event is triggered
    // only when it crosses from positive to negative

}



/*
g++ -std=c++20 -O3 -Iinclude -Iexternal/xdiff/include -Iexternal/xdiff/external/lazy/include -Iexternal/polybox/include -Iexternal/ndspan/include tutorials/RuntimeEvents.cpp -o runtime_events && ./runtime_events
*/