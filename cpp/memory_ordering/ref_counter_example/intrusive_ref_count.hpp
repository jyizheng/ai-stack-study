#pragma once

#include <atomic>
#include <cstddef>
#include <utility>
#include <type_traits>
#include <new>

// Minimal intrusive reference counting primitives demonstrating correct memory ordering.
// - Increments only need atomicity (relaxed).
// - Decrements must publish-with-release and pair with an acquire before destruction,
//   so that the destructor observes all prior writes by the last owner.
//
// This mirrors the core idea used by std::shared_ptr's control block counters,
// though std::shared_ptr also maintains separate strong/weak counts and other logic.

struct RefCounted {
  // Start at 1 when created/returned from a factory like make_intrusive.
  mutable std::atomic<std::size_t> refs{1};

protected:
  // Ensure polymorphic cleanup works.
  virtual ~RefCounted() = default;

public:
  void add_ref() const noexcept {
    // Only atomicity is required for increments; no ordering necessary.
    refs.fetch_add(1, std::memory_order_relaxed);
  }

  // Returns true if this call deleted the object (i.e., ref count reached zero).
  bool release_ref() const noexcept {
    // Use release on the decrement to publish prior writes in the releasing thread(s).
    // If this was the last reference, perform an acquire fence before destruction
    // to synchronize-with the release and make prior writes visible to the destructor.
    if (refs.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      delete this;
      return true;
    }
    return false;
  }

  std::size_t use_count_relaxed() const noexcept {
    // For diagnostics only; not a reliable liveness check under concurrency.
    return refs.load(std::memory_order_relaxed);
  }
};

// A simple intrusive smart pointer that expects T to derive from RefCounted.
// This is intentionally minimal: no weak pointers, no aliasing constructors, etc.
template <class T>
class IntrusivePtr {
  static_assert(std::is_base_of<RefCounted, T>::value,
                "IntrusivePtr<T>: T must derive from RefCounted");

public:
  using element_type = T;

  constexpr IntrusivePtr() noexcept = default;

  // Takes ownership of p (assumes p has refcount >= 1 for this owner).
  explicit IntrusivePtr(T* p) noexcept : ptr_(p) {}

  // Copy: add one reference (relaxed).
  IntrusivePtr(const IntrusivePtr& other) noexcept : ptr_(other.ptr_) {
    if (ptr_) ptr_->add_ref();
  }

  // Move: steal the pointer.
  IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  // Construct from a raw pointer without incrementing if 'add_ref' is false.
  // Useful for factory functions that return a freshly-created object with refs == 1.
  struct adopt_ref_t { explicit adopt_ref_t() = default; };
  static constexpr adopt_ref_t adopt_ref{};

  IntrusivePtr(T* p, adopt_ref_t) noexcept : ptr_(p) {}

  // Copy assign.
  IntrusivePtr& operator=(const IntrusivePtr& other) noexcept {
    if (this == &other) return *this;
    // Add ref first in case other.ptr_ == ptr_ (self-assignment protection)
    T* p = other.ptr_;
    if (p) p->add_ref();
    reset(ptr_);
    ptr_ = p;
    return *this;
  }

  // Move assign.
  IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
    if (this == &other) return *this;
    reset(ptr_);
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~IntrusivePtr() {
    reset(ptr_);
  }

  void reset() noexcept {
    reset(ptr_);
    ptr_ = nullptr;
  }

  void reset(T* p) noexcept {
    if (p == ptr_) return;
    T* old = ptr_;
    ptr_ = p;
    if (old) old->release_ref(); // may delete 'old'
  }

  T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  void swap(IntrusivePtr& other) noexcept { std::swap(ptr_, other.ptr_); }

private:
  T* ptr_ = nullptr;
};

// Convenience factory that constructs T and returns IntrusivePtr<T> adopting the initial ref.
template <class T, class... Args>
IntrusivePtr<T> make_intrusive(Args&&... args) {
  static_assert(std::is_base_of<RefCounted, T>::value,
                "make_intrusive<T>: T must derive from RefCounted");
  T* p = new T(std::forward<Args>(args)...);
  // T's RefCounted base starts with refs == 1, so adopt it.
  return IntrusivePtr<T>(p, typename IntrusivePtr<T>::adopt_ref_t{});
}