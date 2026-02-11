#include "../../../include/pyode/lib_impl/PyResult_impl.hpp"


namespace ode{

//===========================================================================================
//                                      PyOdeResult
//===========================================================================================



PyOdeResult::PyOdeResult(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type): DtypeDispatcher(scalar_type), res(result), q0_shape(q0_shape){}


PyOdeResult::PyOdeResult(const PyOdeResult& other) : DtypeDispatcher(other.scalar_type), q0_shape(other.q0_shape) {
        DISPATCH(void, this->res = other.template cast<T>()->clone();)
}


PyOdeResult::PyOdeResult(PyOdeResult&& other) noexcept : DtypeDispatcher(other.scalar_type), res(other.res), q0_shape(std::move(other.q0_shape)) {
    other.res = nullptr;
}


PyOdeResult::~PyOdeResult(){
        DISPATCH(void, delete cast<T>();)
    res = nullptr;
}

PyOdeResult& PyOdeResult::operator=(const PyOdeResult& other){
    if (&other != this){
        DISPATCH(void, delete cast<T>();)
        DISPATCH(void, this->res = other.template cast<T>()->clone();)
        q0_shape = other.q0_shape;
    }
    return *this;
}


PyOdeResult& PyOdeResult::operator=(PyOdeResult&& other) noexcept{
    if (&other != this){
        DISPATCH(void, delete cast<T>();)
        this->res = other.res;
        q0_shape = other.q0_shape;
        other.res = nullptr;
    }
    return *this;
}


py::object PyOdeResult::t() const{
    return DISPATCH(py::object,
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        return py::cast(View<T>(r->t().data(), r->t().size()));
    )
}

py::object PyOdeResult::q() const{
    return DISPATCH(py::object,
        auto *r = reinterpret_cast<OdeResult<T> *>(this->res);
        auto shape = getShape<size_t>(py::ssize_t(r->t().size()), this->q0_shape);
        return py::cast(View<T>(r->q().data(), shape.data(), shape.size()));
    )
}

py::dict PyOdeResult::event_map() const{
    return DISPATCH(py::object,
        EventMap result = reinterpret_cast<const OdeResult<T>*>(this->res)->event_map();
        return to_PyDict(result);
    )
}

py::tuple PyOdeResult::event_data(const py::str& event) const{
    return DISPATCH(py::object,
        auto* r = reinterpret_cast<OdeResult<T>*>(this->res);
        std::vector<T> t_data = r->t_filtered(event.cast<std::string>());
        Array<T> t_res(t_data.data(), t_data.size());
        Array2D<T, 0, 0> q_data = r->q_filtered(event.cast<std::string>());
        auto shape = getShape<size_t>(py::ssize_t(t_res.size()), this->q0_shape);
        Array<T> q_res(q_data.release(), shape.data(), shape.size(), true);
        return py::make_tuple(py::cast(t_res), py::cast(q_res));
    )
}


py::bool_ PyOdeResult::diverges() const{
    return DISPATCH(py::bool_,
        return py::cast(cast<T>()->diverges());
    )
}

py::bool_ PyOdeResult::success() const{
    return DISPATCH(py::bool_,
        return py::cast(cast<T>()->success());
    )
}

py::float_ PyOdeResult::runtime() const{
    return DISPATCH(py::float_,
        return py::cast(cast<T>()->runtime());
    )
}

py::str PyOdeResult::message() const{
    return DISPATCH(py::str,
        return py::cast(cast<T>()->message());
    )
}

void PyOdeResult::examine() const{
    DISPATCH(void,
        return cast<T>()->examine();
    )
}



//===========================================================================================
//                                      PyOdeSolution
//===========================================================================================


PyOdeSolution::PyOdeSolution(void* result, const std::vector<py::ssize_t>& q0_shape, int scalar_type) : PyOdeResult(result, q0_shape, scalar_type), nsys(prod(q0_shape)) {}

py::object PyOdeSolution::operator()(const py::object& t) const{
    try {
        // Try to convert t to a numpy array
        py::array arr = py::array::ensure(t);
        return DISPATCH(py::object, return this->_get_array<T>(arr);)
    } catch (const py::cast_error&) {
        // If conversion fails, treat as a scalar
        return DISPATCH(py::object, return this->_get_frame<T>(t);)    }
}

//===========================================================================================
//                                EXPLICIT TEMPLATE INSTANTIATIONS
//===========================================================================================


#define DEFINE_ODERESULT(T) \
    template class OdeResult<T, 0>; \
    template class OdeSolution<T, 0>;

DEFINE_ODERESULT(float)
DEFINE_ODERESULT(double)
DEFINE_ODERESULT(long double)
#ifdef MPREAL
DEFINE_ODERESULT(mpfr::mpreal)
#endif

#undef DEFINE_ODERESULT

} // namespace ode