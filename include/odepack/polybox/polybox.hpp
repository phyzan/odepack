#ifndef POLYBOX_HPP
#define POLYBOX_HPP

#include <utility>
#include <cassert>
#include <type_traits>

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace pbox {

template<typename T>
class Box;

///////////////////////////////////////////////////////////////////////////////
// owner<Type>
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
//   - Can move ownership to a Box<Type>
//
///////////////////////////////////////////////////////////////////////////////
template<typename Type>
class owner {
    static_assert(std::is_same_v<Type, std::decay_t<Type>>, "Type cannot be a reference");
    static_assert(!std::is_pointer_v<Type>, "Type cannot be a pointer");

public:

    // Takes ownership of a raw pointer
    FORCE_INLINE explicit owner(Type* object) : ptr(object) {}

    // Constructs Type in-place with forwarded arguments
    // template<typename... Args>
    // explicit owner(Args&&... args) : ptr(new Type(std::forward<Args>(args)...)) {}

    // Default constructs to null
    constexpr owner() = default;

    FORCE_INLINE ~owner() { delete ptr; }

    // Deep copy via clone()
    FORCE_INLINE explicit owner(const owner& other) : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

    template<typename U>
    requires (std::is_convertible_v<U*, Type*>)
    FORCE_INLINE owner(Box<U> box) : ptr(box.release()) {}

    // Move: transfers ownership
    FORCE_INLINE owner(owner&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // Copy assignment: deep copy via clone()
    FORCE_INLINE owner& operator=(const owner& other) {
        if (&other != this) {
            delete ptr;
            ptr = other.ptr ? other.ptr->clone() : nullptr;
        }
        return *this;
    }

    // Move assignment: transfers ownership
    FORCE_INLINE owner& operator=(owner&& other) noexcept {
        if (&other != this) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    // Take ownership from a Box
    FORCE_INLINE owner& operator=(Box<Type>&& box) {
        delete ptr;
        ptr = box.ptr;
        box.ptr = nullptr;
        return *this;
    }

    // Member access (asserts non-null in debug)
    FORCE_INLINE Type* operator->() {
        assert(ptr != nullptr && "dereferencing null owner");
        return ptr;
    }

    FORCE_INLINE const Type* operator->() const {
        assert(ptr != nullptr && "dereferencing null owner");
        return ptr;
    }

    owner& steal(Type* obj){
        assert(obj != ptr && "Attempted to reassign the same pointer");
        delete ptr;
        ptr = obj;
        return *this;
    }

    // Cast and transfer ownership to Box<Derived>.
    // For polymorphic types: uses dynamic_cast, returns null Box on failure.
    // For non-polymorphic types: uses static_cast, always transfers.
    template<typename Derived>
    FORCE_INLINE Box<Derived> move_cast() {
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

    FORCE_INLINE const Type* get() const {
        return ptr;
    }

    // Move ownership to a Box
    FORCE_INLINE Box<Type> move() {
        Box<Type> box(ptr);
        ptr = nullptr;
        return box;
    }

    Type* release() {
        Type* temp = ptr;
        ptr = nullptr;
        return temp;
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
//   - Can only be created by owner via move() or cast(), or make_box()
//   - Can be moved back into a owner
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

    template<typename U>
    requires (std::is_convertible_v<U*, T*>)
    Box(Box<U>&& other) noexcept : ptr(other.ptr) {
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

    FORCE_INLINE operator bool() const {
        return ptr != nullptr;
    }

private:

    template<typename U>
    friend class owner;

    template<typename U>
    friend class Box;

    template<typename U, typename... Args>
    friend Box<U> make_box(Args&&... args);

    // Private: only owner can construct
    FORCE_INLINE explicit Box(T* object) : ptr(object) {}

    // Private: only owner can move
    FORCE_INLINE T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }

    T* ptr = nullptr;
};


template<typename Type, typename... Args>
FORCE_INLINE Box<Type> make_box(Args&&... args) {
    return Box<Type>(new Type(std::forward<Args>(args)...));
}

} // namespace pbox

#endif // POLYBOX_HPP
