#ifndef FINDIFF_HPP
#define FINDIFF_HPP

#include <algorithm>
#include <limits>
#include "../../ndspan/arrays.hpp"

namespace ode{

/**
 * @brief Approximates the Jacobian matrix using finite differences.
 * 
 * @tparam T The type of the elements.
 * @tparam Callable The type of the callable object representing the function.
 * @param f The callable object representing the function.
 * @param out The output (F-storage) array for the Jacobian matrix, of size n x n
 * @param worker A temporary array for intermediate computations, of size 4xn
 * @param t The current time.
 * @param q The current state vector (size n).
 * @param dt The time step or perturbation vector (size n) or nullptr.
 * @param threshold A maximum threshold for determining the step size when dt is nullptr.
 * @param n The size of the state vector (size of the ode system)
 * @note Ideally, we want the threshold parameter to be an array of values, defining a scale value for each
 * component of the scale vector. A component-wise tolerance has not been implemented yet in this library for simplicity.
 */
template<typename T, typename Callable>
constexpr void jac_approx(Callable&& f, T* out, T* worker, const T& t, const T* q, const T* dt, const T& threshold, size_t n){
    const T EPS_SQRT = sqrt(std::numeric_limits<T>::epsilon());

    T* x1 = worker;
    T* x2 = worker + n;
    T* y1 = worker + 2*n;
    T* y2 = worker + 3*n;

    ndspan::copy_array(x1, q, n);
    ndspan::copy_array(x2, q, n);

    for (size_t i = 0; i < n; i++) {
        // Compute step size: use provided dt or compute
        const T abs_qi = abs<T>(q[i]);
        const T h_i = (dt != nullptr) ? dt[i] : EPS_SQRT * std::max<T>(threshold, abs_qi);

        x1[i] = q[i] - h_i;
        x2[i] = q[i] + h_i;
        f(y1, t, x1);
        f(y2, t, x2);
        x1[i] = q[i];
        x2[i] = q[i];

        // Compute Jacobian column using central differences
        T* col = out + i * n;
        T two_h = 2 * h_i;
        for (size_t j = 0; j < n; j++) {
            col[j] = (y2[j] - y1[j]) / two_h;
        }
    }
}

}; // namespace ode


#endif // FINDIFF_HPP