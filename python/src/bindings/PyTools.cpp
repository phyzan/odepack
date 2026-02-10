#include "odetemplates.hpp"

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



} // namespace ode