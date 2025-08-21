#include "intrusive_ref_count.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct Foo : RefCounted {
  int value;

  explicit Foo(int v) : value(v) {
    // Publish initialization before anyone else sees the pointer.
    // Subsequent increments are relaxed; the last decrement will synchronize
    // with this thread before the destructor runs.
  }

  ~Foo() override {
    // The acquire fence in the final release_ref() ensures all prior writes
    // to this object (e.g., 'value' updates) are visible here.
    std::cout << "Foo destroyed, value=" << value << "\n";
  }
};

int main() {
  auto sp = make_intrusive<Foo>(42);

  // Demonstrate concurrent copies and releases.
  constexpr int N = 8;
  constexpr int CopiesPerThread = 10000;

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (int i = 0; i < N; ++i) {
    threads.emplace_back([sp] () mutable {
      // Make and drop many copies; increments are relaxed, decrements are release.
      for (int j = 0; j < CopiesPerThread; ++j) {
        IntrusivePtr<Foo> local = sp; // add_ref (relaxed)
        assert(local->value == 42);
        // local goes out of scope here; release_ref (release, maybe final -> acquire fence)
      }
    });
  }

  for (auto& t : threads) t.join();

  // Change a field; other threads have finished, but this shows ordinary writes.
  sp->value = 7;

  // Drop the last strong reference; if this is the last, destructor runs after acquire fence.
  sp.reset();

  // Give stdout time to flush in some environments.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  return 0;
}