#ifndef POLYWRAPPER_HPP
#define POLYWRAPPER_HPP

#include <utility>
#include <cassert>
#include <type_traits>
#define FORCE_INLINE __attribute__((always_inline)) inline

namespace pbox {

template<typename T>
class Box;

///////////////////////////////////////////////////////////////////////////////
// PolyWrapper<Type>
//
// A cloneable polymorphic value wrapper. Owns a heap-allocated object and
// provides value semantics through a required clone() method.
//
// Requirements:
//   - Type must have: Type* clone() const
//
// Ownership:
//   - Copy: deep copy via clone()
//   - Move: transfers ownership, source becomes null
//   - Can release ownership to a Box<Type>
//
///////////////////////////////////////////////////////////////////////////////
template<typename Type>
class PolyWrapper {
    static_assert(std::is_same_v<Type, std::decay_t<Type>>, "Type cannot be a reference");
    static_assert(!std::is_pointer_v<Type>, "Type cannot be a pointer");

public:

    static PolyWrapper<Type> Null; // Represents a null wrapper
    // Takes ownership of a raw pointer
    FORCE_INLINE explicit PolyWrapper(Type* object) : ptr(object) {}

    // Constructs Type in-place with forwarded arguments
    // template<typename... Args>
    // explicit PolyWrapper(Args&&... args) : ptr(new Type(std::forward<Args>(args)...)) {}

    // Default constructs to null
    constexpr PolyWrapper() = default;

    FORCE_INLINE ~PolyWrapper() { delete ptr; }

    // Deep copy via clone()
    FORCE_INLINE explicit PolyWrapper(const PolyWrapper& other) : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

    FORCE_INLINE PolyWrapper(Box<Type>&& box) : ptr(box.release()) {}

    // Move: transfers ownership
    FORCE_INLINE PolyWrapper(PolyWrapper&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // Copy assignment: deep copy via clone()
    FORCE_INLINE PolyWrapper& operator=(const PolyWrapper& other) {
        if (&other != this) {
            delete ptr;
            ptr = other.ptr ? other.ptr->clone() : nullptr;
        }
        return *this;
    }

    // Move assignment: transfers ownership
    FORCE_INLINE PolyWrapper& operator=(PolyWrapper&& other) noexcept {
        if (&other != this) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    // Take ownership from a Box
    FORCE_INLINE PolyWrapper& operator=(Box<Type>&& box) {
        delete ptr;
        ptr = box.ptr;
        box.ptr = nullptr;
        return *this;
    }

    // Member access (asserts non-null in debug)
    FORCE_INLINE Type* operator->() {
        assert(ptr != nullptr && "dereferencing null PolyWrapper");
        return ptr;
    }

    FORCE_INLINE const Type* operator->() const {
        assert(ptr != nullptr && "dereferencing null PolyWrapper");
        return ptr;
    }

    PolyWrapper& take_ownership(Type* obj){
        assert(obj != ptr && "Attempted to reassign the same pointer");
        delete ptr;
        ptr = obj;
        return *this;
    }

    // Cast and transfer ownership to Box<Derived>.
    // For polymorphic types: uses dynamic_cast, returns null Box on failure.
    // For non-polymorphic types: uses static_cast, always transfers.
    template<typename Derived>
    FORCE_INLINE Box<Derived> release_cast() {
        Type* tmp = ptr;
        if constexpr (std::is_polymorphic_v<Type>) {
            if (Derived* derived = dynamic_cast<Derived*>(tmp)) {
                ptr = nullptr;
                return Box<Derived>(derived);
            }
            return Box<Derived>(nullptr);
        } else {
            ptr = nullptr;
            return Box<Derived>(static_cast<Derived*>(tmp));
        }
    }

    // Non-owning cast for const access (returns raw pointer)
    template<typename Derived>
    FORCE_INLINE const Derived* cast() const {
        return static_cast<const Derived*>(ptr);
    }

    FORCE_INLINE operator bool() const {
        return ptr != nullptr;
    }

    // Release ownership to a Box
    FORCE_INLINE Box<Type> release() {
        Box<Type> box(ptr);
        ptr = nullptr;
        return box;
    }

protected:
    Type* ptr = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// Box<T>
//
// A move-only owning pointer. Similar to std::unique_ptr but with restricted
// construction to enforce ownership safety.
//
// Ownership:
//   - Move-only (no copy)
//   - Can only be created by PolyWrapper via release() or cast()
//   - Can be moved back into a PolyWrapper
//
///////////////////////////////////////////////////////////////////////////////
template<typename T>
class Box {
    static_assert(std::is_same_v<T, std::decay_t<T>>, "T cannot be a reference");
    static_assert(!std::is_pointer_v<T>, "T cannot be a pointer");

public:

    Box() = default;

    // Move constructor: transfers ownership
    FORCE_INLINE Box(Box&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // Move assignment: transfers ownership
    FORCE_INLINE Box& operator=(Box&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    // No copy
    Box(const Box&) = delete;
    Box& operator=(const Box&) = delete;

    FORCE_INLINE ~Box() { delete ptr; }

    // Member access (asserts non-null in debug)
    FORCE_INLINE T* operator->() const {
        assert(ptr != nullptr && "dereferencing null Box");
        return ptr;
    }

private:
    template<typename U>
    friend class PolyWrapper;

    // Private: only PolyWrapper can construct
    FORCE_INLINE explicit Box(T* object) : ptr(object) {}

    // Private: only PolyWrapper can release
    FORCE_INLINE T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }

    T* ptr = nullptr;
};


template<typename Type, typename... Args>
FORCE_INLINE PolyWrapper<Type> make(Args&&... args) {
    return PolyWrapper<Type>(new Type(std::forward<Args>(args)...));
}

template<typename Type>
PolyWrapper<Type> PolyWrapper<Type>::Null = PolyWrapper<Type>();

} // namespace pbox

#endif // POLYWRAPPER_HPP
