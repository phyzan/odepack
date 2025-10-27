
#ifndef TENSORS_HPP
#define TENSORS_HPP


#include <utility>
#include <array>
#include <cstring>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <iostream>

#define THIS static_cast<Derived*>(this)
#define THIS_C static_cast<const Derived*>(this)
#define CONST_CAST(TYPE, FUNC) const_cast<TYPE>(static_cast<const CLS*>(this)->FUNC);

#define STATIC_IDX_ASSERT(idx, ND) static_assert(sizeof...(idx) == ND, "Incorrect number of indices")

#define DYNAMIC_IDX_ASSERT(idx, nd) assert(sizeof...(idx) == ND && "Incorrect number of indices")

#define POS_IDX_ASSERT(idx, ND) assert(((idx > 0) && ...) && "All shape dims must be positive")

#define CF_LAYOUT(L) (L == Layout::C || L == Layout::F)

#define DEFAULT_RULE_OF_FOUR(CLASSNAME)                  \
    CLASSNAME(const CLASSNAME& other) = default;      \
    CLASSNAME(CLASSNAME&& other) = default;           \
    CLASSNAME& operator=(const CLASSNAME& other) = default; \
    CLASSNAME& operator=(CLASSNAME&& other) = default;

enum class Layout : std::uint8_t { C, F, Z};

template<typename... Ts>
concept INT_T = (std::is_integral_v<Ts>  && ...);

template<typename T>
inline void copy_array(T* dest, const T* src, size_t size){
    if (size==0) {return;}
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<typename T, size_t size>
inline void copy_array(T* dest, const T* src){
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<typename T>
inline bool equal_arrays(const T* a, const T* b, size_t size){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}

template<typename T>
std::string to_string(const T& value, int digits = 3) {
    static_assert(std::is_arithmetic<T>::value || std::is_class<T>::value, "T must be a numeric or class type with ostream << defined");

    std::ostringstream out;
    out << std::setprecision(digits) << std::scientific << value;
    return out.str();
}

template<typename T>
std::string array_repr(T* array, size_t size, int digits) {
    std::string result;
    if (size == 0){
        return result;
    }
    for (size_t i=0; i<size-1; i++){
        result += to_string(array[i], digits) + " ";
    }
    return result + to_string(array[size-1], digits);
}

template<typename T, size_t size>
inline bool equal_arrays(const T* a, const T* b){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}


template<size_t... Args>
size_t _validate_size(size_t size){
    constexpr size_t expected = (Args * ...);
    assert(size == expected && "Invalid initializer list size");
    return size;
}


template<typename T, Layout L, size_t... DIMS>
class NdSpan{

static_assert((L != Layout::Z) || (... && (((DIMS & (DIMS - 1)) == 0))), "Z-order layout supported only for dimensions that are a power of 2");

public:

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t SHAPE[ND > 0 ? ND : 1] = {DIMS...};
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));

    NdSpan() = default;

    template<INT_T... Args>
    explicit NdSpan(Args... args){
        POS_IDX_ASSERT(args, ND);
        this->reshape(args...);
    }

    inline constexpr size_t size() const{
        return N;
    }

    inline constexpr size_t ndim() const{
        return ND;
    }

    inline const size_t* shape() const {
        return SHAPE;
    }

    inline constexpr size_t shape(size_t i) const {
        return SHAPE[i];
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const noexcept {
        STATIC_IDX_ASSERT(idx, ND);
        if constexpr (L == Layout::F || L == Layout::C){
            return _static_offset_impl(std::index_sequence_for<Idx...>{}, idx...);
        }
        else if constexpr (L == Layout::Z){
            return _morton_impl(idx...);
        }
        return 0;
    }

    template<INT_T... Idx>
    inline constexpr size_t morton_index(Idx... idx) const noexcept{
        STATIC_IDX_ASSERT(idx, ND);
        return _morton_impl(idx...);
    }

    template<INT_T... Args>
    inline void reshape(Args... shape){
        if (! ((shape==DIMS) && ...)){
            throw std::runtime_error("Runtime dims do not match template dims in Tensor constructor");
        }
    }

    template<INT_T... Args>
    inline void constexpr resize(Args... shape){
        //can only resize with the exact same shape
        reshape(shape...);
    }

private:

    template<size_t... I, INT_T... Idx>
    inline constexpr size_t _static_offset_impl(std::index_sequence<I...>, Idx... idx) const noexcept {
        return ((static_cast<size_t>(idx) * STRIDES[I]) + ...);
    }

protected:
    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        // Only for 'C' or 'F' order
        size_t stride = 1;
        if constexpr (L == Layout::C) {
            for (size_t i=nd; i-- > 0;){
                s[i] = stride;
                stride *= shape[i];
            }
        }
        else if constexpr (L == Layout::F) {
            for (size_t i = 0; i < nd; ++i) {
                s[i] = stride;
                stride *= shape[i];
            }
        }
    }

    static constexpr std::array<size_t, ND> make_strides(){
        std::array<size_t, ND> s{};
        if constexpr (ND > 0){
            set_strides(s, SHAPE, ND);
        }
        return s;
    }

    template<typename ShapeType>
    static constexpr size_t get_bits(const ShapeType& shape, size_t nd){
        size_t res = 0;
        for (size_t i = 0; i < nd; ++i) {
            size_t n = shape[i] - 1;
            size_t bits = 0;
            while (n >> bits) {++bits;}
            res = std::max(bits, res);
        }
        return res;
    }

    inline static constexpr std::array<size_t, ND> STRIDES = make_strides();
    inline static constexpr size_t BITS = get_bits(SHAPE, ND);

    template<size_t bit, size_t... Axis, INT_T... Coords>
    inline constexpr size_t _morton_dims_loop(std::index_sequence<Axis...>, Coords... coords) const {
        return ( (((static_cast<size_t>(coords) >> bit) & size_t(1)) << (bit*ND + Axis)) | ... );
    }

    template<size_t... bit, size_t... AXIS, INT_T... Coords>
    inline constexpr size_t _morton_bit_loop(std::index_sequence<bit...>, std::index_sequence<AXIS...>, Coords... coords) const {
        return ( _morton_dims_loop<bit>(std::make_index_sequence<ND>{}, coords...) | ... );
    }

    template<INT_T... Coords>
    inline constexpr size_t _morton_impl(Coords... coords) const {
        return _morton_bit_loop(std::make_index_sequence<BITS>{}, std::make_index_sequence<ND>{}, coords...);
    }
};


template<typename T, Layout L, size_t... DIMS>
class DynamicNdSpan : public NdSpan<T, L, DIMS...>{

    using Base = NdSpan<T, L, DIMS...>;
    using CLS = DynamicNdSpan<T, L, DIMS...>;

    inline static constexpr size_t ND = sizeof...(DIMS);
    inline static constexpr size_t N = (ND == 0 ? 0 : (DIMS * ... * 1));

public:

    DynamicNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr DynamicNdSpan(Args... shape) : _n(sizeof...(shape) == 0 ? 0 : (shape*...*1)) {
        this->reshape(shape...);
    }

    //COPY CONSTRUCTORS

    DynamicNdSpan(const DynamicNdSpan& other) requires (ND>0) = default;

    DynamicNdSpan(const DynamicNdSpan& other) requires (ND==0) : Base(other), _nd(other._nd), _n(other._n){
        _dyn_shape = other._nd > 0 ? new size_t[other._nd] : nullptr;
        copy_array(_dyn_shape, other._dyn_shape, _nd);
        if constexpr (CF_LAYOUT(L)){
            _dyn_strides = other._nd > 0 ? new size_t[other._nd] : nullptr;
            copy_array(_dyn_strides, other._dyn_strides, _nd);
        }
    }

    //MOVE CONSTRUCTORS
    DynamicNdSpan(DynamicNdSpan&& other) requires (ND>0) = default;

    DynamicNdSpan(DynamicNdSpan&& other) noexcept requires (ND==0) : Base(other), _nd(other._nd), _n(other._n), _dyn_shape(other._dyn_shape), _dyn_strides(other._dyn_strides) {
        other._dyn_shape = nullptr;
        other._dyn_strides = nullptr;
    }

    //ASSIGNMENT OPERATORS
    DynamicNdSpan& operator=(const DynamicNdSpan&) requires (ND>0) = default;

    DynamicNdSpan& operator=(const DynamicNdSpan& other) requires (ND==0){
        if (&other != this){
            Base::operator=(other);
            if (_nd != other._nd){
                _nd = other._nd;
                delete[] _dyn_shape;
                if constexpr (CF_LAYOUT(L)) {delete[] _dyn_strides;}
                if (_nd > 0){
                    _dyn_shape = new size_t[_nd];
                    if constexpr (CF_LAYOUT(L)) {_dyn_strides = new size_t[_nd];}
                }
                else{
                    _dyn_shape = nullptr;
                    if constexpr (CF_LAYOUT(L)) {_dyn_strides = nullptr;}
                }
            }
            _n = other._n;
            copy_array(_dyn_shape, other._dyn_shape, _nd);
            if constexpr (CF_LAYOUT(L)) {copy_array(_dyn_strides, other._dyn_strides, _nd);}

        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS
    DynamicNdSpan& operator=(DynamicNdSpan&&) requires (ND>0) = default;

    DynamicNdSpan& operator=(DynamicNdSpan&& other) noexcept requires (ND==0){
        if (&other != this){
            Base::operator=(std::move(other));
            if constexpr (CF_LAYOUT(L)) {delete[] _dyn_strides;}
            delete[] _dyn_shape;
            _n = other._n;
            _nd = other._nd;
            _dyn_shape = other._dyn_shape;
            if constexpr (CF_LAYOUT(L)) {_dyn_strides = other._dyn_strides;}
            other._n = 0;
            other._nd = 0;
            other._dyn_shape = nullptr;
            if constexpr (CF_LAYOUT(L)) {other._dyn_strides = nullptr;}
        }
        return *this;
    }

    ~DynamicNdSpan() {
        if constexpr (ND==0){
            delete[] _dyn_shape;
            if constexpr (CF_LAYOUT(L)) {delete[] _dyn_strides;}
            _dyn_shape = nullptr;
            if constexpr (CF_LAYOUT(L)) {_dyn_strides = nullptr;}
        }
    }



    template<INT_T... Args>
    void constexpr resize(Args... shape){

        //TODO make sure non of these are zero at runtime, and they are equal to the template parameter
        constexpr size_t new_nd = sizeof...(shape);
        size_t new_dims[new_nd] = {static_cast<size_t>(shape)...};
        static_assert(new_nd > 0, "Cannot call resize() with no arguments");

        if constexpr (ND==0){
            POS_IDX_ASSERT(shape, new_nd);
            if (new_nd != _nd){
                if (new_nd > _nd){
                    //only reallocate in this case
                    delete[] _dyn_shape;
                    _dyn_shape = new size_t[new_nd]{ static_cast<size_t>(shape)... };
                    if constexpr (CF_LAYOUT(L)) {
                        delete[] _dyn_strides;
                        _dyn_strides = new size_t[new_nd];
                    }
                }
                else{
                    size_t i=0;
                    ((_dyn_shape[i++] = shape), ...);
                }
                _nd = new_nd;
            }
            else{
                copy_array(_dyn_shape, new_dims, _nd);
            }
        }
        else if constexpr (N==0){
            //ND > 0, but some template dims are zero
            static_assert((new_nd == ND), "Constructor must be called with as many arguments as the number of template parameters");
            POS_IDX_ASSERT(shape, ND);
            for (size_t i=0; i<ND; i++){
                if ((Base::SHAPE[i] > 0) && new_dims[i] != Base::SHAPE[i]){
                    throw std::runtime_error("Runtime dims do not match template dims in Tensor constructor");
                }
                _fixed_shape[i] = new_dims[i];
            }
        }
        else{
            Base::reshape(new_dims);
        }

        if constexpr (N==0) {
            _n = (shape*...);
            size_t* s = this->_strides();
            if constexpr (CF_LAYOUT(L)) {this->set_strides(s, this->shape(), this->ndim());}
            else if constexpr (L == Layout::Z){
                _bits = this->get_bits(this->shape(), this->ndim());
            }
        }
    }

    template<INT_T... Args>
    void constexpr reshape(Args... shape){
        const size_t new_size = (shape*...);
        if (new_size != _n){
            throw std::runtime_error("Invalid new shape. The total size of the tensor is not conserved");
        }
        return this->resize(shape...);
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const {
        if constexpr (N > 0){
            return Base::offset(idx...);
        }
        else if constexpr (ND > 0){
            STATIC_IDX_ASSERT(idx, ND);
        }
        else if constexpr (ND == 0){
            DYNAMIC_IDX_ASSERT(idx, this->ndim());
        }
        return _offset_impl(std::index_sequence_for<Idx...>{}, idx...);
    }

    inline size_t size() const{
        if constexpr (N > 0) {return N;}
        else {return _n;}
    }

    inline size_t ndim() const{
        if constexpr (ND>0){
            return ND;
        }
        else{
            return _nd;
        }
    }

    inline const size_t* shape() const {
        if constexpr (ND>0){
            return _fixed_shape;
        }
        else{
            return _dyn_shape;
        }
    }

    inline size_t shape(size_t i) const {
        if constexpr (ND>0){
            return _fixed_shape[i];
        }
        else{
            return _dyn_shape[i];
        }
    }
    
private:

    inline size_t* _strides(){
        if constexpr (ND == 0){
            return _dyn_strides;
        }
        else{
            return _fixed_strides.data();
        }
    }

    template<size_t... I, INT_T... Idx>
    inline constexpr size_t _offset_impl(std::index_sequence<I...>, Idx... idx) const noexcept {
        if constexpr (CF_LAYOUT(L)){
            if constexpr (ND > 0){
                return ((static_cast<size_t>(idx) * _fixed_strides[I]) + ...);
            }
            else{
                return ((static_cast<size_t>(idx) * _dyn_strides[I]) + ...);
            }
        }
        else if constexpr (L == Layout::Z){
            return _morton_runtime(idx...);
        }
        return 0; //code should not reach this
    }

    template<INT_T... Idx>
    inline constexpr size_t _morton_runtime(Idx... idx) const {
        if constexpr (N > 0){
            return Base::morton_index(idx...);
        }
        return _dyn_morton_impl(idx...);
    }

    template<size_t... Axis, INT_T... Coords>
    inline constexpr size_t _dyn_morton_dims_loop(size_t bit, std::index_sequence<Axis...>, Coords... coords) const {
        return ( (((static_cast<size_t>(coords) >> bit) & size_t(1)) << (bit*sizeof...(Axis) + Axis)) | ... );
    }

    template<INT_T... Coords>
    inline constexpr size_t _dyn_morton_impl(Coords... coords) const {
        size_t result = 0;
        for (size_t bit = 0; bit < _bits; ++bit) {
            result |= _dyn_morton_dims_loop(bit, std::make_index_sequence<sizeof...(coords)>{}, coords...);
        }
        return result;
    }

    size_t  _nd = ND;
    size_t  _n = (ND == 0 ? 0 : (DIMS * ... * 1));
    size_t* _dyn_shape = nullptr;
    size_t* _dyn_strides = nullptr;
    size_t  _fixed_shape[ND>0 ? ND : 1] = {DIMS...}; //used when ND > 0, but N==0
    std::array<size_t, ND> _fixed_strides = Base::make_strides();//used when ND > 0
    size_t _bits = 0; //used only when L == Layout::Z and N == 0

};





template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractNdView : public std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), NdSpan<T, L, DIMS...>, DynamicNdSpan<T, L, DIMS...>>{

    using CLS = AbstractNdView<Derived, L, T, DIMS...>;
    using Base = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), NdSpan<T, L, DIMS...>, DynamicNdSpan<T, L, DIMS...>>;

protected:

    using Base::Base;

public:

    inline const T* data() const{
        //override
        return THIS_C->data();
    }

    inline T* data() {
        return CONST_CAST(T*, data());
    }

    inline const T* begin() const{ return data();}

    inline T* begin() {return data();}

    inline const T* end() const {return data()+size();}

    inline T* end() {return data()+size();}

    template<INT_T... Idx>
    inline constexpr const T& operator()(Idx... idx) const noexcept {
        return data()[this->offset(idx...)];
    }

    template<INT_T... Idx>
    inline constexpr T& operator()(Idx... idx) noexcept {
        return data()[this->offset(idx...)];
    }

    inline const T& operator[](size_t i) const{
        return data()[i];
    }

    inline T& operator[](size_t i){
        return data()[i];
    }


    // function definitions that only assist C++ parsers

    inline constexpr size_t size() const{
        return Base::size();
    }

    inline constexpr size_t ndim() const{
        return Base::ndim();
    }

    inline const size_t* shape() const {
        return Base::shape();
    }

    inline constexpr size_t shape(size_t i) const {
        return Base::shape(i);
    }

    template<INT_T... Idx>
    inline constexpr size_t offset(Idx... idx) const noexcept {
        return Base::offset(idx...);
    }

    template<INT_T... Args>
    inline void reshape(Args... shape){
        return Base::reshape(shape...);
    }

    template<INT_T... Args>
    inline void resize(Args... shape){
        return Base::resize(shape...);
    }

};




template<typename T, Layout L, size_t... DIMS>
class NdView : public AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>{

    using Base = AbstractNdView<NdView<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit NdView(T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<INT_T... Args>
    explicit NdView(T* data, Args... shape) : Base(shape...), _data(data) {}

    inline const T* data() const{
        return _data;
    }

    inline T* data() {
        return _data;
    }

    void set_data(T* data) {
        _data = data;
    }

private:

    T* _data;
};




template<typename T, Layout L, size_t... DIMS>
class Tensor : public AbstractNdView<Tensor<T, L, DIMS...>, L, T, DIMS...>{

    using CLS = Tensor<T, L, DIMS...>;
    using Base = AbstractNdView<Tensor<T, L, DIMS...>, L, T, DIMS...>;

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t ND = sizeof...(DIMS);

public:

    inline static constexpr bool IS_HEAP = ((N==0) || (sizeof(T)*N > 80000));

    Tensor() {
        if constexpr (IS_HEAP && N > 0) {
            _dynamic_arr = new T[N];
        }
    }

    explicit Tensor(const T* arr) requires (N>0) {
        if constexpr (IS_HEAP) {
            _dynamic_arr = new T[N];
        }
        copy_array<T, N>(data(), arr);
    }
    
    template<INT_T... Args>
    explicit Tensor(Args... shape) : Base(shape...) {
        static_assert(sizeof...(shape) > 0, "Cannot construct shape from no shape");
        if constexpr (IS_HEAP) {
            if (this->size() > 0){
                _dynamic_arr = new T[this->size()];
            }
        }
    }

    template<INT_T... Args>
    explicit Tensor(const T* arr, Args... shape) : Tensor(shape...){
        copy_array(this->data(), arr, this->size());
    }

    Tensor(std::initializer_list<T> array) requires (ND<2 && (N == 0)) : Tensor(array.begin(), array.size()) {}

    Tensor(std::initializer_list<T> array) requires (N>0) : Tensor(array.begin(), (_validate_size<DIMS...>(array.size()), DIMS)...) {}

    //COPY CONSTRUCTORS
    Tensor(const Tensor& other) : Base(other), _dynamic_arr((IS_HEAP && other.size() > 0) ? new T[other.size()] : nullptr) {
        copy_array(data(), other.data(), this->size());
    }

    //MOVE CONSTRUCTORS
    Tensor(Tensor&& other) noexcept : Base(std::move(other)), _dynamic_arr(IS_HEAP ? other._dynamic_arr : nullptr) {
        if constexpr (IS_HEAP) { other._dynamic_arr = nullptr;}
        else{
            copy_array(data(), other.data(), this->size());
        }
    }

    //ASSIGNMENT OPERATORS
    Tensor& operator=(const Tensor& other) {
        if (&other != this){
            if constexpr (IS_HEAP){
                if (this->size() != other.size()){
                    delete[] _dynamic_arr;
                    _dynamic_arr = other.size() > 0 ? new T[other.size()] : nullptr;                
                }
            }
            Base::operator=(other);
            copy_array(data(), other.data(), this->size());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATORS

    Tensor& operator=(Tensor&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            if constexpr (IS_HEAP){
                delete[] _dynamic_arr;
                _dynamic_arr = other._dynamic_arr;
                other._dynamic_arr = nullptr;
            }
            else{
                copy_array(data(), other.data(), this->size());
            }
        }
        return *this;
    }

    Tensor& set(const T& value){
        size_t s = this->size();
        T* d = this->data();
        #pragma omp simd
        for (size_t i=0; i<s; i++){
            d[i] = value;
        }
        return *this;
    }

    ~Tensor() {
        if constexpr (IS_HEAP){
            delete[] _dynamic_arr;
            _dynamic_arr = nullptr;
        }
    }

    inline const T* data() const{
        if constexpr (IS_HEAP){
            return _dynamic_arr;
        }
        else{
            return _fixed_arr;
        }
    }

    inline T* data() {
        if constexpr (IS_HEAP){
            return _dynamic_arr;
        }
        else{
            return _fixed_arr;
        }
    }

    template<INT_T... Size>
    void resize(Size... newsize){
        size_t current_size = this->size();
        Base::resize(newsize...);
        if constexpr (IS_HEAP){
            size_t total_size = (newsize * ...);
            if (total_size != current_size){
                delete[] _dynamic_arr;
                if (total_size == 0){
                    _dynamic_arr = nullptr;
                }
                else{
                    _dynamic_arr = new T[total_size];
                }
                
            }
        }
        
    }

    inline std::string repr(int digits=8) const {
        return array_repr(this->data(), this->size(), digits);
    }

protected:

    T* _dynamic_arr = nullptr;
    T _fixed_arr[IS_HEAP ? 1 : N];
};



template<typename T, Layout L, size_t N=0>
class Tensor1D : public Tensor<T, L, N>{

public:
    using Tensor<T, L, N>::Tensor;

};

template<typename T, Layout L, size_t Nr, size_t Nc>
class Tensor2D : public Tensor<T, L, Nr, Nc>{

public:
    using Tensor<T, L, Nr, Nc>::Tensor;

    inline size_t Nrows() const {return this->shape()[0];}

    inline size_t Ncols() const {return this->shape()[1];}

    std::string repr(int digits=8) const {
        std::vector<size_t> column_str_len(this->Ncols());
        for (size_t j=0; j<this->Ncols(); j++){
            size_t len = to_string((*this)(0, j), digits).size();
            for (size_t i=0; i<this->Nrows(); i++){
                len = std::max(len, to_string((*this)(i, j), digits).size());
            }
            column_str_len[j] = len;
        }
        
        std::string result, element;
        for (size_t i=0; i<this->Nrows(); i++){
            for (size_t j=0; j<this->Ncols(); j++){
                element = to_string((*this)(i, j), digits);
                result += std::string(column_str_len[j]-element.size(), ' ') + element;
                if (j < this->Ncols()-1){
                    result += " ";
                }
            }
            if (i<this->Nrows()-1){
                result += "\n";
            }
        }

        return result;
    }
};


//alias
template<typename T, size_t N=0>
using Array1D = Tensor1D<T, Layout::C, N>;

template<typename T, size_t Nr=0, size_t Nc=0>
using Array2D = Tensor2D<T, Layout::C, Nr, Nc>;


template<typename T, size_t M, size_t N>
void mat_dot_vec(T* res, const T* mat, const T* vec){
    for (size_t i=0; i<M; i++){
        T sum = 0;
        for (size_t j=0; j<N; j++){
            sum += mat[i*N + j] * vec[j];
        }
        res[i] = sum;
    }
}


/*
TODO

Handle resizing when Layout == Z
Handle arrays when and maybe allow any shape when layout == Z
Instead of Layout as template parameter, put an Options template parameter:
There, we choose Layout, when to reallocate when resizing whether to perform shape checks,...
*/


#endif