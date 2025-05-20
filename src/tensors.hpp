#ifndef TENSORS_HPP
#define TENSORS_HPP

#include <initializer_list>
#include "vectorized.hpp"
#include "operation.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>


using std::pow, std::sqrt;

size_t prod(const size_t* k, const size_t& len){
    size_t res = 1;
    for (size_t i=0; i<len; i++){
        res *= k[i];
    }
    return res;
}

template<class S>
void _apply_dot(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n);


class Shape{

public:
    Shape(const size_t* size, const int& ndim) : _ndim(ndim){
        _init(size);
    }

    Shape(std::initializer_list<size_t> size) : _ndim(size.size()){
        _init(size.begin());
    }

    Shape(const Shape& other): _ndim(other._ndim){
        _init(other._size);
    }

    Shape(Shape&& other): _ndim(other._ndim), _size(other._size){}

    Shape& operator=(const Shape& other){

        if (&other == this){
            return *this;
        }
        _ndim = other._ndim;
        delete[] _size;
        _init(other._size);
        return *this;
    }

    bool operator==(const Shape& other) const{
        if (_ndim != other._ndim){
            return false;
        }
        for (int i=0; i<_ndim; i++){
            if (_size[i] != other._size[i]){
                return false;
            }
        }
        return true;
    }

    ~Shape(){
        delete[] _size;
        _size = nullptr;
    }

    inline const size_t& operator[](const int& axis) const{
        return _size[axis];
    }

    inline size_t& operator[](const int& axis){
        return _size[axis];
    }

    inline const int& ndim() const{
        return _ndim;
    }

    inline size_t total_size() const{
        size_t s = 1;
        for (int i=0; i<_ndim; i++){
            s *= _size[i];
        }
        return s;
    }

    size_t ravel_index_C(const std::vector<size_t>& indices)const{
        size_t res = 0;
        size_t _prod = 1;
        for (size_t k=_ndim-1; k>0; k--){
            res += indices[k]*_prod;
            _prod *= _size[k];
        }
        return res;
    }

    size_t ravel_index_F(const std::vector<size_t>& indices)const{
        size_t res = 0;
        size_t _prod = 1;
        for (size_t k=0; k<_ndim; k++){
            res += indices[k]*_prod;
            _prod *= _size[k];
        }
        return res;
    }

    std::vector<size_t> unravel_index_C(size_t index) const {
        std::vector<size_t> res(_ndim);
        for (size_t i = 0; i < _ndim; i++) {
            size_t stride = prod(_size + i + 1, _ndim - i - 1);  // size_t* + offset, length
            res[i] = index / stride;
            index %= stride;
        }
        return res;
    }

    std::vector<size_t> unravel_index_F(size_t index) const {
        std::vector<size_t> res(_ndim);
        for (size_t i = 0; i < _ndim; i++) {
            size_t stride = prod(_size, i);  // product of dimensions before axis i
            res[i] = index / stride;
            index %= stride;
        }
        return res;
    }

private:
    size_t* _size;
    int _ndim;

    void _init(const size_t* size){
        _size = new size_t[_ndim];
        for (int i=0; i<_ndim; i++){
            _size[i] = size[i];
        }
    }
};


class Iterator{

public:

    Iterator(const Shape& shape): _shape(shape), _max(shape.total_size()), _state(shape.ndim(), 0){}

    Iterator(const size_t* size, const int& ndim): _shape(size, ndim), _max(prod(size, ndim)), _state(ndim, 0){}

    Iterator(Iterator&& other): _shape(std::move(other._shape)), _max(other._max), _state(std::move(other._state)), _current(other._current){}

    inline const int& ndim() const{
        return _shape.ndim();
    }

    inline const size_t& index(const int& axis) const{
        return _state[axis];
    }

    inline const size_t& max() const{
        return _max;
    }

    inline const size_t& current() const{
        return _current;
    }

    void advance(){
        for (int i=ndim()-1; i>-1; i--){
            if (_state[i] < _shape[i]-1){
                _state[i]++;
                _current++;
                return;
            }
            else{
                _state[i] = 0;
            }
        }
        _current = 0;
    }


private:

    Shape _shape;
    size_t _max;
    std::vector<size_t> _state;
    size_t _current = 0;


};


template<class S>
class Tensor{

public:

    Tensor(std::initializer_list<S> array) : _shape({array.size()}) {
        _init_array(array.begin(), array.size());
    }

    template<typename... Dims, typename = std::enable_if_t<(std::is_convertible_v<Dims, size_t> && ...)>>
    Tensor(Dims... dims) : _shape{static_cast<size_t>(dims)...} {
        _size = _shape.total_size();
        _array = new S[_size];
    }

    Tensor(const Tensor<S>& other): _shape(other._shape){
        _init_array(other._array, other._size);
    }

    Tensor(Tensor<S>&& other): _array(other._array), _size(other._size), _shape(std::move(other._shape)){}

    Tensor<S>& operator=(const Tensor<S>& other){
        delete[] _array;
        _shape = other._shape;
        _init_array(other._array, other._size);
        return *this;
    }

    Tensor<S>& operator+=(const Tensor<S>& other){
        _check_shape(other);
        for (size_t i = 0; i < _size; i++){
            _array[i] += other._array[i];
        }
        return *this;
    }

    Tensor<S>& operator+=(const S& other){
        for (size_t i = 0; i < _size; i++){
            _array[i] += other;
        }
        return *this;
    }

    Tensor<S>& operator-=(const Tensor<S>& other){
        _check_shape(other);
        for (size_t i = 0; i < _size; i++){
            _array[i] -= other._array[i];
        }
        return *this;
    }

    Tensor<S>& operator-=(const S& other){
        for (size_t i = 0; i < _size; i++){
            _array[i] -= other;
        }
        return *this;
    }

    Tensor<S>& operator*=(const Tensor<S>& other){
        _check_shape(other);
        for (size_t i = 0; i < _size; i++){
            _array[i] *= other._array[i];
        }
        return *this;
    }

    Tensor<S>& operator*=(const S& other){
        for (size_t i = 0; i < _size; i++){
            _array[i] *= other;
        }
        return *this;
    }

    Tensor<S>& operator/=(const Tensor<S>& other){
        _check_shape(other);
        for (size_t i = 0; i < _size; i++){
            _array[i] /= other._array[i];
        }
        return *this;
    }

    Tensor<S>& operator/=(const S& other){
        for (size_t i = 0; i < _size; i++){
            _array[i] /= other;
        }
        return *this;
    }

    Tensor<S>& apply_add(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]+b[i];
        }
        return *this;
    }

    Tensor<S>& apply_add(const Tensor<S>& a, const S& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]+b;
        }
        return *this;
    }

    Tensor<S>& apply_sub(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]-b[i];
        }
        return *this;
    }

    Tensor<S>& apply_sub(const Tensor<S>& a, const S& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]-b;
        }
        return *this;
    }

    Tensor<S>& apply_mul(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]*b[i];
        }
        return *this;
    }

    Tensor<S>& apply_mul(const Tensor<S>& a, const S& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]*b;
        }
        return *this;
    }

    Tensor<S>& apply_div(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]/b[i];
        }
        return *this;
    }

    Tensor<S>& apply_div(const Tensor<S>& a, const S& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a[i]/b;
        }
        return *this;
    }

    Tensor<S>& apply_div(const S& a, const Tensor<S>& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = a/b[i];
        }
        return *this;
    }

    Tensor<S>& apply_pow(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = pow(a[i], b[i]);
        }
        return *this;
    }

    Tensor<S>& apply_pow(const Tensor<S>& a, const S& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = pow(a[i], b);
        }
        return *this;
    }

    Tensor<S>& apply_pow(const S& a, const Tensor<S>& b){
        _check_shape(a);
        for (size_t i = 0; i < _size; i++){
            _array[i] = pow(a, b[i]);
        }
        return *this;
    }

    Tensor<S>& set_abs(){
        for (size_t i = 0; i < _size; i++){
            _array[i] = (_array[i] < 0) ? -_array[i] : _array[i];
        }
        return *this;
    }

    Tensor<S>& apply_max(const Tensor<S>& a, const Tensor<S>& b){
        _check_shape(a);
        _check_shape(b);
        for (size_t i = 0; i < _size; i++){
            _array[i] = (a[i] > b[i]) ? a[i] : b[i];
        }
        return *this;
    }

    S norm_squared() const{
        S res = 0;
        for (size_t& i=0; i<_size; i++){
            res += _array[i]*_array[i];
        }
        return res;
    }

    inline S norm() const{
        return sqrt(norm_squared());
    }

    inline S rms_norm() const{
        return sqrt(norm_squared()/_size);
    }

    bool allFinite() const {
        for (size_t i = 0; i < _size; ++i) {
            if (std::isnan(_array[i]) || std::isinf(_array[i])) {
                return false;
            }
        }
        return true;
    }

    inline const S& operator[](const size_t& index) const {
        return _array[index];
    }

    inline S& operator[](const size_t& index) {
        return _array[index];
    }

    template <typename... Indices>
    std::size_t operator()(Indices... args) const {
        static_assert((std::is_convertible_v<Indices, std::size_t> && ...), "All indices must be size_t-convertible");
        return _array[_shape.ravel_index_C(static_cast<std::size_t>(args)...)];
    }

    ~Tensor() {
        delete[] _array;
        _array = nullptr;
    }

    inline const size_t& size() const{
        return _size;
    }

    inline const int& ndim() const{
        return _shape.ndim();
    }

    inline const Shape& shape() const{
        return _shape;
    }

    Tensor<S>& reshape(const std::vector<size_t>& shape){
        if (prod(shape.data(), shape.size()) != _size){
            throw std::runtime_error("Invalid Tensor shape");
        }
        else{
            _shape = Shape(shape.data(), shape.size());
        }
        return *this;
    }

    inline void apply_dot_product(const Tensor<S>& a, const Tensor<S>& b){
        _apply_dot(_array, a._array, b._array, _shape[0], a._shape[1], b._shape[1]);
    }


private:

    S* _array;
    size_t _size;
    Shape _shape;

    void _init_array(const S* data, const size_t& size){
        _array = new S[size];
        _size = size;
        assign_elementwise(_array, data, size);
    }

    void _check_shape(const Tensor<S>& other) const{
        if (_shape != other._shape){
            throw std::runtime_error("Incompatible shapes");
        }
    }

};


template<class S>
void _apply_dot(S* r, const S* a, const S* b, const size_t& m, const size_t& s, const size_t& n){
    // #pragma omp simd
    for (size_t k=0; k<m*n; k++){
        size_t i = k/n;
        size_t j = k % n;
        S _sum = 0;
        for (size_t q=0; q<s; q++){
            _sum += a[i*s + q] * b[q*n + j];
        }
        r[i*n+j] = _sum;
    }
}


#endif