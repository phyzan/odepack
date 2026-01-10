#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "arrays.hpp"
#include "iterators.hpp"

template<typename T, size_t N, size_t M>
constexpr bool same_elements(const std::array<T, N>& a, const std::array<T, M>& b){
    if constexpr (N!=M){
        return false;
    }
    bool used[N] = {};

    for (size_t i = 0; i < N; ++i) {
        bool found = false;
        for (size_t j = 0; j < N; ++j) {
            if (!used[j] && a[i] == b[j]) {
                used[j] = true;
                found = true;
                break;
            }
        }
        if (!found){
            return false;
        }
    }
    return true;
}

template<typename T, size_t M, size_t N>
constexpr std::array<T, M+N> joined_arrays(const std::array<T, M>& a, const std::array<T, N>& b){
    std::array<T, M+N> res{};
    for (size_t i=0; i<M; i++){
        res[i] = a[i];
    }
    for(size_t i=0; i<N; i++){
        res[i+M] = b[i];
    }
    return res;
}

template<typename T, size_t M>
constexpr std::array<T, M> joined_arrays(const std::array<T, M>& a){
    return a;
}

template<typename T, size_t M, size_t... N>
requires (sizeof...(N)>1)
constexpr std::array<T, (N+...+M)> joined_arrays(const std::array<T, M>& a, const std::array<T, N>&... b){
    return joined_arrays(a, joined_arrays(b...));
}


template<typename ArrayType>
constexpr bool all_equal(const ArrayType& array){
    for (size_t i=0; i<array.size()-1; i++){
        if (array[i] != array[i+1]){
            return false;
        }
    }
    return true;
}



template<typename T>
struct ItemCounter {

    template<T... x>
    struct CountHelper {
        // Helper: check if a value is already in a pack
        template<T value, T... ys>
        static constexpr bool contains() {
            return ((value == ys) || ...); // fold expression
        }

        // Recursive helper to count uniques
        template<T first, T... rest>
        static constexpr size_t count() {
            if constexpr (contains<first, rest...>()) {
                return count<rest...>();
            } else {
                if constexpr (sizeof...(rest) == 0) {
                    return 1;
                } else {
                    return 1 + count<rest...>();
                }
            }
        }

        static constexpr size_t value = []() constexpr {
            if constexpr (sizeof...(x) == 0){ return 0;}
            else{ return count<x...>();}
        }();
    };

    template<T... x>
    static constexpr size_t unique_item_count() {
        return CountHelper<x...>::value;
    }

    template<T... x>
    static constexpr auto unique_items(){
        constexpr size_t M = unique_item_count<x...>();
        std::array<T, M> result{};
        std::array<T, sizeof...(x)> items = {x...};

        size_t k = 0;
        for (size_t i = 0; i < sizeof...(x); ++i) {
            bool seen = false;
            for (size_t j = 0; j < i; ++j) {
                if (items[i] == items[j]) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                result[k++] = items[i];
            }
        }
        return result;
    }
};

template<typename T , size_t N>
constexpr size_t unique_item_count(const std::array<T, N>& items){
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j) {
            if (items[i] == items[j]) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            ++count;
        }
    }
    return count;
}

template<typename T, size_t N>
constexpr size_t item_count(const T& x, const std::array<T, N>& items){
    size_t res = 0;
    for (size_t i=0; i<N; i++){
        if (items[i]==x){
            res++;
        }
    }
    return res;
}

template<typename T, size_t N>
constexpr bool contains(const T& x, const std::array<T, N>& items){
    for (size_t i=0; i<N; i++){
        if (items[i]==x){
            return true;
        }
    }
    return false;
}

template<char label>
struct Index{
    static constexpr char Label = label;

    template<char x>
    constexpr bool operator==(Index<x>) const {
        return x==label;
    }
};


template<typename T>
struct IsIndex : std::false_type{};

template<char x>
struct IsIndex<Index<x>> : std::true_type{};

template<typename T>
concept IdxLike = std::is_integral_v<T> || IsIndex<T>::value;

template<typename T>
concept AnyIdx = IsIndex<T>::value;

template<char y, char... x>
constexpr size_t char_count(){
    return ((y==x)+...+0);
}

template<size_t y, size_t... x>
constexpr size_t int_count(){
    return ((y==x)+...+0);
}


template<char... label>
struct IndexSequence{

    static constexpr size_t ND = sizeof...(label);

    static constexpr std::array<char, ND> labels = {label...};

    static_assert(((item_count(label, labels)==1) && ...), "Index labels must be unique in IndexSequence");

    constexpr IndexSequence(Index<label>...) requires (sizeof...(label) > 0) {}
    constexpr IndexSequence() = default;

    template<char x>
    static constexpr bool contains(Index<x>){
        return ((label==x) || ...);
    }

    template<char x>
    static constexpr size_t place(Index<x>){
        static_assert(contains(Index<x>()));
        for (size_t I=0; I<ND; I++){
            if (x==labels[I]){
                return I;
            }
        }
        assert(false && "Invalid loop");
        return ND;
    }
};

template<typename T>
struct IsIdxSequence : std::false_type{};

template<char... label>
struct IsIdxSequence<IndexSequence<label...>> : std::true_type {};

template<typename T>
concept IdxSequence = IsIdxSequence<T>::value;

struct VirtualIndex{
    size_t position;
    char label;
};


template<typename ViewType, IdxLike... IdxType>
struct VirtualTensor;

template<typename T>
struct IsVirtualTensor : std::false_type{};

template<typename ViewType, IdxLike... IdxType>
struct IsVirtualTensor<VirtualTensor<ViewType, IdxType...>> : std::true_type{};

template<typename T>
concept VTensor = IsVirtualTensor<T>::value;

template<typename Scalar, VTensor... Tensor>
struct IntermediateNetwork : std::tuple<Tensor...>{

    static_assert((std::is_same_v<Scalar, typename Tensor::Scalar>&&...), "All tensors must have the same numeric type");

    IntermediateNetwork(Tensor... data) : std::tuple<Tensor...>(data...){}

    IntermediateNetwork(Scalar a, Tensor... data) : std::tuple<Tensor...>(data...), c(a){}

    Scalar c = 1;
};

template<typename ViewType, IdxLike... IdxType>
struct VirtualTensor{

    static constexpr size_t ND = sizeof...(IdxType);

    using Viewer = ViewType;

    using Scalar = typename ViewType::value_type;

    template<size_t I>
    using ElemType = std::tuple_element_t<I, std::tuple<IdxType...>>;

    VirtualTensor(ViewType* array, IdxType... idx) : array(array), idx_data(idx...) {

        if constexpr (ViewType::ND > 0){
            static_assert(ViewType::ND==ND, "Invalid number of indices");
        }
        else{
            assert(array->ndim()==ND && "Invalid number of indices");
        }

        assert(_valid_idx() && "Invalid repeated labels corresponding to axis of different length");
    }

    VirtualTensor(const VirtualTensor& other) = default;
    VirtualTensor(VirtualTensor&& other) noexcept = default;
    VirtualTensor& operator=(VirtualTensor&& other) noexcept {
        return operator=(static_cast<const VirtualTensor&>(other));
    }

    template<size_t I>
    static constexpr bool is_virtual_idx(){
        return !std::is_integral_v<ElemType<I>>;
    }

    static constexpr bool is_valid_lhs(){
        return same_elements(idx_labels(), unique_labels());
    }

    INLINE VirtualTensor& operator=(const VirtualTensor& other){
        if (other.array->data() == this->array->data()){
            assert(other.idx_data == this->idx_data && "Invalid operation: Identical array lhs=rhs with different contraction guide");
            return *this;
        }
        _contract_over(other);
        return *this;
    }

    template<VTensor Tensor>
    INLINE VirtualTensor& operator=(const Tensor& other){
        assert(other.array->data() != this->array->data() && "Invalid operation: Identical array lhs=rhs with different contraction guide");
        _contract_over(other);
        return *this;
    }

    template<VTensor... Tensors>
    INLINE VirtualTensor& operator=(const IntermediateNetwork<Scalar, Tensors...>& other){
        _contract_over(other);
        return *this;
    }

    template<typename IdxMap, size_t I, INT_T... IntType>
    INLINE size_t tensor_idx(IntType... idx_pack) const{

        if constexpr (std::is_integral_v<ElemType<I>>) {
            //The I-th element is an integer, so just return it
            return std::get<I>(idx_data);
        }
        else{
            //ElemType is Index<char>, so return the corresponding element in the pack.
            static_assert(IdxMap::place(ElemType<I>())<sizeof...(idx_pack));
            return pack_elem<IdxMap::place(ElemType<I>())>(idx_pack...);
        }
    }

    template<typename IdxMap, INT_T... IntType>
    INLINE const Scalar& operator()(IntType... idx) const{

        return _call_aux<IdxMap>(std::make_index_sequence<ND>(), idx...);
    }

    template<typename IdxMap, INT_T... IntType>
    INLINE Scalar& operator()(IntType... idx){

        return _call_aux<IdxMap>(std::make_index_sequence<ND>(), idx...);
    }

    static constexpr size_t label_count(){
        return ((!std::is_integral_v<IdxType>)+...);
    }

    static constexpr auto idx_labels(){
        constexpr size_t n = label_count();
        std::array<char, n> res{};
        size_t k=0;
        FOR_LOOP(size_t, I, ND,
            if constexpr (is_virtual_idx<I>()) {
                res[k++] = ElemType<I>::Label;
            }
        );
        return res;
    }

    static constexpr size_t unique_label_count(){
        return unique_item_count(idx_labels());
    }

    static constexpr auto unique_labels(){
        constexpr auto items = idx_labels();
        return EXPAND(size_t, label_count(), I,
            return ItemCounter<char>::unique_items<items[I]...>();
        );
    }

    template<char label>
    static constexpr size_t first_idx_of(Index<label>){
        static_assert(contains(Index<label>()));
        size_t i=0;
        bool found = false;
        FOR_LOOP(size_t, I, ND,
            if constexpr (is_virtual_idx<I>()) {
                if (!found && ElemType<I>::Label==label) {
                    i = I;
                    found = true;
                }
            }
        );
        return i;
    }

    template<char label>
    static constexpr bool contains(Index<label>){
        return ::contains(label, idx_labels());
    }

    constexpr size_t shape(size_t i) const{
        return array->shape(i);
    }
    
    template<char label>
    constexpr size_t shape_of(Index<label>) const{
        constexpr size_t I = first_idx_of(Index<label>());
        return shape(I);
    }

    static constexpr auto virtual_idx(){
        std::array<VirtualIndex, label_count()> res{};
        size_t i=0;
        FOR_LOOP(size_t, I, label_count(),
            if constexpr (is_virtual_idx<I>()) {
                res[i++] = {I, ElemType<I>::Label};
            }
        );
        return res;
    }

    Scalar contract() const{
        return make_tensor_network<Scalar>(IndexSequence<>(), *this).contract_all();
    }

    ViewType* array;
    std::tuple<IdxType...> idx_data;

private:

    template<typename IdxMap, size_t... I, INT_T... IntType>
    INLINE const Scalar& _call_aux(std::index_sequence<I...>, IntType... idx) const{
        return static_cast<const ViewType*>(array)->operator()(tensor_idx<IdxMap, I>(idx...)...);
    }

    template<typename IdxMap, size_t... I, INT_T... IntType>
    INLINE Scalar& _call_aux(std::index_sequence<I...>, IntType... idx){
        return array->operator()(tensor_idx<IdxMap, I>(idx...)...);
    }

    constexpr bool _valid_idx() const{
        constexpr std::array<VirtualIndex, label_count()> virt_idx = virtual_idx();
        for (size_t i=0; i<virt_idx.size(); i++){
            for (size_t j=i+1; j<virt_idx.size(); j++){
                if (virt_idx[i].label==virt_idx[j].label){
                    if (array->shape(virt_idx[i].position) != array->shape(virt_idx[j].position)){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template<VTensor OtherType>
    INLINE void _contract_over(const OtherType& other){
        static_assert(is_valid_lhs(), "Tensor is not eligible for LHS of tensor contraction (repeated indices)");

        constexpr auto free_idx_tmp = idx_labels();
        constexpr auto free_idx = EXPAND(size_t, free_idx_tmp.size(), I,
            return IndexSequence(Index<free_idx_tmp[I]>()...);
        );
        make_tensor_network<Scalar>(free_idx, other).contract(*this);
    }

    template<VTensor... Tensors>
    INLINE void _contract_over(const IntermediateNetwork<Scalar, Tensors...>& other){
        static_assert(is_valid_lhs(), "Tensor is not eligible for LHS of tensor contraction (repeated indices)");
        constexpr auto free_idx_tmp = idx_labels();
        constexpr auto free_idx = EXPAND(size_t, free_idx_tmp.size(), I,
            return IndexSequence(Index<free_idx_tmp[I]>()...);
        );
        EXPAND(size_t, sizeof...(Tensors), I,
            assert(((std::get<I>(other).array->data() != this->array->data())&&...) && "Lhs tensor data occurs in the rhs of the contraction");
            make_tensor_network<Scalar>(other.c, free_idx, std::get<I>(other)...).contract(*this);
        );
    }
};

template<typename Scalar, IdxSequence FreeIdx, VTensor... Tensor>
class TensorNetwork{

    //Each Tensor object is callable(full_idx_pack), if it is provided with an IndexSequence
    //that maps each index of the pack to a Index<char>. Then, after passing the entire full_idx_pack
    //into the call operator of the Tensor, the tensor extract the indices it needs.

    static constexpr size_t NodeCount = sizeof...(Tensor);
    static constexpr size_t flat_label_count = (Tensor::label_count()+...);
    static constexpr size_t unique_label_count = unique_item_count(joined_arrays(Tensor::idx_labels()...));

    static constexpr size_t Nlegs = FreeIdx::ND;

    static constexpr size_t Ncontr = unique_label_count-Nlegs;

    static constexpr std::array<char, Nlegs> free_idx = FreeIdx::labels;

public:

    TensorNetwork(FreeIdx free, Tensor... A) : nodes(A...){
        assert(_valid_labels() && "Repeated index labels correspond to different axis size");
    }

    TensorNetwork(Scalar coef, FreeIdx free, Tensor... A) : nodes(A...), coef(coef){
        assert(_valid_labels() && "Repeated index labels correspond to different axis size");
    }

    TensorNetwork() = delete;

    static constexpr std::array<char, flat_label_count> flat_idx_labels(){
        return joined_arrays(Tensor::idx_labels()...);
    }

    static constexpr auto flat_idx_unique_labels(){
        return joined_arrays(Tensor::unique_labels()...);
    }

    static constexpr std::array<char, unique_label_count> unique_idx_labels(){
        constexpr auto items = joined_arrays(Tensor::idx_labels()...);
        return EXPAND(size_t, items.size(), I,
            return ItemCounter<char>::unique_items<items[I]...>();
        );
    }

    template<char label>
    static constexpr size_t node_count(Index<label>){
        size_t res = 0;
        constexpr auto arr = flat_idx_unique_labels();
        for (char name : arr){
            if (name == label){
                res++;
            }
        }
        return res;
    }

    template<char label>
    static constexpr size_t occur_count(Index<label>){
        size_t res = 0;
        constexpr auto arr = flat_idx_labels();
        for (char name : arr){
            if (name == label){
                res++;
            }
        }
        return res;
    }

    template<char label>
    static constexpr size_t contains(Index<label> x){
        return occur_count(x) > 0;
    }

    template<char label>
    INLINE constexpr size_t shape_of(Index<label> x) const{
        static_assert(contains(x), "Tensor network does not contain index");

        return _shape_finder<0>(x);
    }

    constexpr std::array<size_t, Nlegs> shape_free() const{
        std::array<size_t, Nlegs> res{};

        FOR_LOOP(size_t, I, Nlegs,
            res[I] = shape_of(Index<free_idx[I]>());
        );
        return res;
    }

    constexpr std::array<size_t, Ncontr> shape_contr() const{
        std::array<size_t, Ncontr> res{};

        FOR_LOOP(size_t, I, Ncontr,
            res[I] = shape_of(Index<contr_idx[I]>());
        );
        return res;
    }

    static constexpr std::array<char, Ncontr> contracted_idx(){
        std::array<char, Ncontr> res{};
        size_t k=0;
        for (char name : unique_idx_labels()){
            if (!::contains(name, free_idx)){
                res[k++] = name;
            }
        }
        return res;
    }

    template<size_t I, IdxSequence Seq, INT_T... IntType>
    INLINE Scalar get(Seq map, IntType... i) const{
        return std::get<I>(nodes).template operator()<Seq>(i...);
    }

    template<IdxSequence Seq, INT_T... IntType>
    INLINE Scalar get_all(Seq map, IntType... i) const{
        return EXPAND(size_t, NodeCount, I,
                return (std::get<I>(nodes).template operator()<Seq>(i...)*...);
            );
    }
    
    template<VTensor LhsType>
    INLINE void contract(LhsType lhs) const requires (FreeIdx::ND>0){

        //static_assert that lhs has the same free indices as this class
        static_assert(_valid_lhs<LhsType>(), "Invalid lhs indices");

        auto free_map = EXPAND(size_t, Nlegs, I,
            return IndexSequence(Index<free_idx[I]>()...);
        );

        auto contr_map = EXPAND(size_t, Ncontr, I,
            return IndexSequence(Index<contr_idx[I]>()...);
        );

        return _contract_aux(lhs, free_map, contr_map);
    }

    Scalar contract_all() const {

        //static_assert that lhs has the same free indices as this class

        auto contr_map = EXPAND(size_t, unique_label_count, I,
            return IndexSequence(Index<unique_idx_labels()[I]>()...);
        );

        Scalar sum = 0;

        auto f = [&](auto... i) LAMBDA_INLINE{
            sum += this->get_all(contr_map, i...);
        };

        auto iter = EXPAND(size_t, unique_label_count, I,
            return DynamicNDIterator<unique_label_count>(shape_of(Index<unique_idx_labels()[I]>())...);
        );
        iter.iterate(f);
        return sum*coef;
    }

    static constexpr std::array<char, Ncontr> contr_idx = contracted_idx();

    template<size_t I>
    using TensorType = std::tuple_element_t<I, std::tuple<Tensor...>>;

private:

    template<VTensor LhsType>
    static constexpr bool _valid_lhs(){
        return same_elements(free_idx, LhsType::idx_labels());
    }

    template<typename LhsViewType, IdxLike... IdxType, char... If, char... Ic>
    INLINE void _contract_aux(VirtualTensor<LhsViewType, IdxType...> lhs, IndexSequence<If...>, IndexSequence<Ic...>) const{

        auto shape_fr = shape_free();
        auto shape_co = shape_contr();
        DynamicNDIterator<Nlegs> free_iter = EXPAND(size_t, Nlegs, I,
            return DynamicNDIterator<Nlegs>(shape_fr[I]...);
        );

        using LhsIdxMap = IndexSequence<If...>;
        using FullIdxMap = IndexSequence<If..., Ic...>;

        if constexpr (Ncontr>0) {
            DynamicNDIterator<Ncontr> contr_iter = EXPAND(size_t, Ncontr, I,
                return DynamicNDIterator<Ncontr>(shape_co[I]...);
            );

            auto free_func = [&](auto... i) LAMBDA_INLINE {
                Scalar sum = 0;
                auto inner_func = [&](auto... j) LAMBDA_INLINE {
                    sum += this->get_all(FullIdxMap(), i..., j...);
                };

                contr_iter.iterate(inner_func);
                lhs.template operator()<LhsIdxMap>(i...) = sum*coef;
            };
            free_iter.iterate(free_func);
        }else{
            auto free_func = [&](auto... i) LAMBDA_INLINE {
                lhs.template operator()<LhsIdxMap>(i...) = this->get_all(FullIdxMap(), i...)*coef;
            };
            free_iter.iterate(free_func);
        }
    }

    template<char label>
    constexpr auto _shapes_of(Index<label> x) const {
        constexpr size_t Nc = node_count(x);
        std::array<size_t, Nc> res{};
        size_t i=0;
        FOR_LOOP(size_t, I, NodeCount, 
            if constexpr (TensorType<I>::contains(x)) {
                res[i++] = std::get<I>(nodes).shape_of(x);
            }
        );
        return res;
    }

    constexpr bool _valid_labels() const{
        constexpr auto uniques = unique_idx_labels();
        bool valid = true;
        FOR_LOOP(size_t, I, unique_label_count,
            auto shapes = _shapes_of(Index<uniques[I]>());
            if (!all_equal(shapes)){
                valid = false;
            }
        );
        return valid;
    }

    static constexpr bool _valid_free(){
        constexpr auto labels = flat_idx_labels();

        //make sure all free indices are unique
        for (size_t i=0; i<FreeIdx::ND; i++){
            if (item_count(FreeIdx::labels[i], FreeIdx::labels)!=1){
                return false;
            }
        }

        //make sure each of the free is found at least once in the tensor network
        for (size_t i=0; i<FreeIdx::ND; i++){
            if (item_count(FreeIdx::labels[i], labels)==0){
                return false;
            }
        }

        return true;
    }

    template<size_t I, char label>
    INLINE constexpr size_t _shape_finder(Index<label> x) const{
        if constexpr (TensorType<I>::contains(x)) {
            return std::get<I>(nodes).shape_of(x);
        }
        else{
            return _shape_finder<I+1>(x);
        }
    }

    static_assert(_valid_free(), "Invalid free indices");

    std::tuple<Tensor...> nodes;
    Scalar coef=1;

};

template<typename T, typename FreeIdx, typename... Tensors>
TensorNetwork<T, FreeIdx, Tensors...>
make_tensor_network(FreeIdx free, Tensors... tensors){
    return {free, tensors...};
}

template<typename T, typename FreeIdx, typename... Tensors>
TensorNetwork<T, FreeIdx, Tensors...>
make_tensor_network(T coef, FreeIdx free, Tensors... tensors){
    return {coef, free, tensors...};
}

template<VTensor A, VTensor B>
requires (std::is_same_v<typename A::Scalar, typename B::Scalar>)
INLINE auto operator*(A a, B b){
    return IntermediateNetwork<typename A::Scalar, A, B>(a, b);
}

template<typename Scalar, VTensor A, typename... Tensor>
INLINE auto operator*(A a, IntermediateNetwork<Scalar, Tensor...> b){
    return EXPAND(size_t, sizeof...(Tensor), I,
        return IntermediateNetwork<Scalar, A, Tensor...>(a, std::get<I>(b)...);
    );
}

template<typename Scalar, VTensor B, typename... Tensor>
INLINE auto operator*(IntermediateNetwork<Scalar, Tensor...> a, B b){
    return EXPAND(size_t, sizeof...(Tensor), I,
        return IntermediateNetwork<Scalar, Tensor..., B>(std::get<I>(a)..., b);
    );
}

template<typename Scalar, typename... TensorA, typename... TensorB>
INLINE auto operator*(IntermediateNetwork<Scalar, TensorA...> a, IntermediateNetwork<Scalar, TensorB...> b){
    return EXPAND(size_t, sizeof...(TensorA), I,
        return EXPAND(size_t, sizeof...(TensorB), J,
            return IntermediateNetwork<Scalar, TensorA..., TensorB...>(a.c*b.c, std::get<I>(a)..., std::get<J>(b)...);
        );
    );
}




template<typename Scalar, VTensor B>
requires (std::is_same_v<Scalar, typename B::Scalar>)
INLINE auto operator*(Scalar a, B b){
    return IntermediateNetwork<Scalar, B>(a, b);
}

template<typename Scalar, VTensor A>
requires (std::is_same_v<Scalar, typename A::Scalar>)
INLINE auto operator*(A a, Scalar b){
    return IntermediateNetwork<Scalar, A>(b, a);
}


template<typename Scalar, typename... Tensor>
INLINE auto operator*(IntermediateNetwork<Scalar, Tensor...> a, Scalar b){
    return EXPAND(size_t, sizeof...(Tensor), I,
        return IntermediateNetwork<Scalar, Tensor...>(a.c*b, std::get<I>(a)...);
    );
}

template<typename Scalar, typename... Tensor>
INLINE auto operator*(Scalar a, IntermediateNetwork<Scalar, Tensor...> b){
    return EXPAND(size_t, sizeof...(Tensor), I,
        return IntermediateNetwork<Scalar, Tensor...>(a*b.c, std::get<I>(b)...);
    );
}




template<typename Array, IdxLike... Int>
requires (!std::is_integral_v<Int>||...)
inline auto tensor_call(const Array& x, Int... idx){
    return VirtualTensor(x, idx...);
}

template<typename Array, IdxLike... Int>
requires (!std::is_integral_v<Int>||...)
inline auto tensor_call(Array& x, Int... idx){
    return VirtualTensor(x, idx...);
}

/*
TODO

contract closed networks first
find optimal contraction ordering
add vectorization
*/

#endif