#include "../../../include/pyode/lib_impl/PyTools_impl.hpp"
#include "../../../include/ode/Interpolation/StateInterp_impl.hpp"

namespace ode{

//===========================================================================================
//                                      DtypeDispatcher
//===========================================================================================


DtypeDispatcher::DtypeDispatcher(const std::string& dtype_){
    this->scalar_type = DTYPE_MAP.at(dtype_);
}

DtypeDispatcher::DtypeDispatcher(int dtype_) : scalar_type(dtype_) {}


//===========================================================================================
//                                      Helper Functions
//===========================================================================================

template<>
 std::vector<py::ssize_t> getShape(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape){
    std::vector<py::ssize_t> result;
    result.reserve(1 + shape.size()); // Pre-allocate memory for efficiency
    result.push_back(dim1);        // Add the first element
    result.insert(result.end(), shape.begin(), shape.end()); // Append the original vector
    return result;
}

 std::vector<py::ssize_t> shape(const py::object& obj) {
    py::array arr = py::array::ensure(obj);
    const ssize_t* shape_ptr = arr.shape();  // Pointer to shape data
    auto ndim = static_cast<size_t>(arr.ndim());  // Number of dimensions
    std::vector<py::ssize_t> res(shape_ptr, shape_ptr + ndim);
    return res;
}

 py::dict to_PyDict(const EventMap& _map){
    py::dict py_dict;
    for (const auto& [key, vec] : _map) {
        py::array_t<size_t> np_array(static_cast<py::ssize_t>(vec.size()), vec.data()); // Create NumPy array
        py_dict[key.c_str()] = np_array; // Assign to dictionary
    }
    return py_dict;
}

 std::vector<EventOptions> to_Options(const py::iterable& d) {
    std::vector<EventOptions> result;

    for (const py::handle& item : d) {
        auto opt = py::cast<EventOptions>(item);
        result.emplace_back(opt);
    }
    result.shrink_to_fit();
    return result;
}


bool is_sorted(const py::array_t<double>& arr){
    const double* ptr = arr.data();
    for (ssize_t i = 1; i < arr.size(); ++i){
        if (ptr[i] <= ptr[i-1]){
            return false;
        }
    }
    return true;
}


// Explicit instantiations for Python callback helpers.
#define DEFINE_PYTOOLS(T) \
    template class Interval<T>; \
    template class Interpolator<T, 0>; \
    template class LocalInterpolator<T, 0>; \
    template class StandardLocalInterpolator<T, 0>; \
    template class LinkedInterpolator<T, 0, Interpolator<T, 0>>; \
    template class StepSequence<T>; \
    template StepSequence<T> to_step_sequence<T>(const py::object& t_eval); \
    template std::vector<T> subvec<T>(const std::vector<T>& x, size_t start, size_t size); \
    template void lin_interp(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, size_t size); \
    template void coef_mat_interp<T>(T* result, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* coef_mat, size_t order, size_t size); \
    template void inv_mat_row_major(T *out, const T *mat, size_t N, T *work, size_t *pivot); \
    template bool all_are_finite(const T *data, size_t n); \
    
#define DEFINE_BUILTIN_SCALAR_ONLY(T) \
    template void py_rhs<T>(T* res, const T& t, const T* q, const T*, const void* obj); \
    template void py_jac<T>(T* res, const T& t, const T* q, const T*, const void* obj); \
    template void py_mask<T>(T* res, const T& t, const T* q, const T*, const void* obj); \
    template T py_event<T>(const T& t, const T* q, const T*, const void* obj); \
    template const py::array_t<T>& PyStruct::get_array<T>() const; \
    template py::array_t<T>& PyStruct::get_array<T>(); \

template bool allEqual(const int *a, const int *b, size_t n);
template std::vector<size_t> getShape<size_t>(const py::ssize_t& dim1, const std::vector<py::ssize_t>& shape);

DEFINE_PYTOOLS(float)
DEFINE_PYTOOLS(double)
DEFINE_PYTOOLS(long double)
#ifdef MPREAL
DEFINE_PYTOOLS(mpfr::mpreal)
#endif

DEFINE_BUILTIN_SCALAR_ONLY(float)
DEFINE_BUILTIN_SCALAR_ONLY(double)
DEFINE_BUILTIN_SCALAR_ONLY(long double)



#undef DEFINE_BUILTIN_SCALAR_ONLY


#undef DEFINE_PYTOOLS


} // namespace ode