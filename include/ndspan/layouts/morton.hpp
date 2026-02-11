#pragma once

#include <unordered_map>
#include <array>
#include "../layouts/standard_layouts.hpp"
#include "../ndspan.hpp"

namespace ndspan{

template<std::integral INT>
constexpr INT next_pow_of_2(INT x){
    INT y = 1;
    while (y < x){
        y *= 2;
    }
    return y;
}

template<size_t... DIMS>
class ZorderNdSpan : public DerivedNdSpan<ZorderNdSpan<DIMS...>, DIMS...>{

    using Base = DerivedNdSpan<ZorderNdSpan<DIMS...>, DIMS...>;

public:

    using Base::Base;

    ZorderNdSpan() = default;

    template<INT_T... Args>
    explicit constexpr ZorderNdSpan(Args... shape) : Base(shape...), _strided_span(shape...){
        this->resize(shape...);
    }

    DEFAULT_RULE_OF_FOUR(ZorderNdSpan)

    INLINE size_t padded_size() const{
        return _padded_size;
    }

    template<INT_T... Args>
    void constexpr resize(Args... shape){
        Base::resize(shape...); //now the shape and size are updated
        size_t new_shape[sizeof...(shape)] = {static_cast<size_t>(shape)...};
        _strided_span.resize(shape...);
        _bits = morton_bits(new_shape, sizeof...(shape));
        _padded_size = padded_size(shape...);

        _id_at_offset.clear();
        _id_at_offset.shrink_to_fit();

        _offset_at_id.clear();
        _offset_at_id.shrink_to_fit();

        if (this->size() > 0){
            //resize maps
            _id_at_offset.resize(this->size());
            _offset_at_id.resize(this->size());

            //temporary idx array to fill its values below
            std::array<size_t, sizeof...(shape)> idx;
            size_t I, true_idx = 0;
            for (size_t padded_offset=0; padded_offset<_padded_size; padded_offset++){
                //retrieve the (i, j, k) index of the padded grid
                this->morton_unpack_idx_from_array(padded_offset, idx);

                //check if we are inside the true grid
                if (!(I=0) && ((idx[I++] < shape)&&...)){
                    //means we are inside the true grid

                    //retrieve id
                    size_t id = _strided_span.offset(idx);

                    //set values in the id<->offset maps
                    _id_at_offset[true_idx] = id;
                    _offset_at_id[id] = true_idx;
                    true_idx++;
                }
            }
        }
    }

    template<INT_T... Idx>
    INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        size_t id = _strided_span.offset(idx...);
        return _offset_at_id[id];
    }

    template<INT_T Idx>
    INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        size_t id = _strided_span.getOffset(idx_ptr);
        return _offset_at_id[id];
    }

    template<INT_T... Idx>
    INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept{
        size_t id = _id_at_offset[offset];
        _strided_span.unpack_idx(id, idx...);
    }

    template<typename ShapeType>
    static constexpr size_t morton_bits(const ShapeType& shape, size_t nd){
        size_t res = 0;
        for (size_t i = 0; i < nd; ++i) {
            size_t n = shape[i] - 1;
            size_t bits = 0;
            while (n >> bits) {++bits;}
            res = std::max(bits, res);
        }
        return res;
    }

    template<INT_T... Args>
    INLINE static constexpr size_t padded_size(Args... shape){
        static_assert(sizeof...(shape) > 0, "The padded_size function requires at least one argument");
        if (((shape == 0)||...)){
            return 0;
        }
        size_t tmp, p = 1;
        size_t shape_arr[sizeof...(shape)] = {size_t(shape)...};
        for (size_t i=0; i<sizeof...(shape);i++){
            tmp = next_pow_of_2(shape_arr[i]);
            p = std::max(p, tmp);
        }
        return ((shape*0+p)*...);
    }

    template<std::integral INT, size_t Nd>
    INLINE constexpr size_t morton_offset(const std::array<INT, Nd>& idx) const {
        return _morton_offset_aux(idx, std::make_index_sequence<Nd>());
    }

    template<INT_T... Coords>
    inline constexpr size_t morton_index(Coords... idx) const {
        this->_dim_check(idx...);
        this->_bounds_check(std::make_index_sequence<sizeof...(idx)>(), idx...);
        size_t result = 0;
        for (size_t bit = 0; bit < _bits; ++bit) {
            result |= _dyn_morton_dims_loop(bit, std::make_index_sequence<sizeof...(idx)>{}, idx...);
        }
        return result;
    }

    template<INT_T... Args>
    inline constexpr void morton_unpack_idx(size_t offset, Args&... idx) const {
        //same as unpack_idx, but directly computes the indices instead of performing memory lookups.
        // offset is the morton index
        ((idx = 0), ...);
        for (size_t bit = 0; bit < _bits; ++bit) {
            _morton_axis_aux<sizeof...(idx)>(offset, bit, std::make_index_sequence<sizeof...(idx)>(), idx...);
        }
    }

    template<size_t Nd, INT_T... Args>
    inline constexpr void morton_unpack_idx_from_array(size_t offset, std::array<size_t, Nd>& idx) const {
        return _morton_unpack_idx_from_array_aux<Nd>(offset, idx, std::make_index_sequence<Nd>());
    }

private:

    template<size_t Axis, size_t Nd>
    INLINE static constexpr size_t
    _morton_bit(size_t morton_index, size_t bit) {
        return (( (morton_index >> (bit * Nd + Axis)) & 1) << bit );
    }

    template<size_t Nd, INT_T... Idx, size_t... I>
    INLINE static constexpr void _morton_axis_aux(size_t offset, size_t bit, std::index_sequence<I...>, Idx&... idx) {
        ((idx |= _morton_bit<I, Nd>(offset, bit)),...);
    }

    template<size_t... Axis, INT_T... Coords>
    INLINE static constexpr size_t _dyn_morton_dims_loop(size_t bit, std::index_sequence<Axis...>, Coords... coords) {
        return size_t(( (((coords >> bit) & 1) << (bit*sizeof...(Axis) + Axis)) | ... ));
    }

    template<std::integral INT, size_t Nd, size_t... I>
    inline constexpr size_t _morton_offset_aux(const std::array<INT, Nd>& idx, std::index_sequence<I...>) const {
        return morton_index(idx[I]...);
    }

    template<size_t Nd, size_t... I, INT_T... Args>
    inline constexpr void _morton_unpack_idx_from_array_aux(size_t offset, std::array<size_t, Nd>& idx, std::index_sequence<I...>) const {
        return this->morton_unpack_idx(offset, idx[I]...);
    }

    size_t _padded_size;
    size_t _bits;
    RowMajorSpan<DIMS...> _strided_span; //converts the index (i, j, ...) to a unique id (integer). The unique id continuously ranges from 0 up to (dim1*dim2*...*dimNd - 1)
    std::vector<size_t> _id_at_offset; 
    std::vector<size_t> _offset_at_id;
};

} // namespace ndspan
