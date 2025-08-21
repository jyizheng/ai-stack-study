// If an atomic store in thread A is tagged memory_order_release,
// an atomic load in thread B from the same variable is tagged memory_order_acquire,
// and the load in thread B reads a value written by the store in thread A,
// then the store in thread A synchronizes-with the load in thread B.

// All memory writes (including non-atomic and relaxed atomic) that happened-before
// the atomic store from the point of view of thread A, become visible side-effects in thread B.
// That is, once the atomic load is completed, thread B is guaranteed to see everything thread A wrote to memory.
// This promise only holds if B actually returns the value that A stored,
// or a value from later in the release sequence.

// The synchronization is established only between the threads releasing and acquiring
// the same atomic variable. Other threads can see different order of memory accesses
// than either or both of the synchronized threads.

// On strongly-ordered systems — x86, SPARC TSO, IBM mainframe, etc. — release-acquire ordering
// is automatic for the majority of operations. No additional CPU instructions are issued
// for this synchronization mode; only certain compiler optimizations are affected
// (e.g., the compiler is prohibited from moving non-atomic stores past the atomic store-release
// or performing non-atomic loads earlier than the atomic load-acquire).
// On weakly-ordered systems (ARM, Itanium, PowerPC), special CPU load or memory fence instructions are used.

// Mutual exclusion locks, such as std::mutex or atomic spinlock, are an example of release-acquire synchronization:
// when the lock is released by thread A and acquired by thread B, everything that took place in the critical section
// (before the release) in the context of thread A has to be visible to thread B (after the acquire)
// which is executing the same critical section.


#include <atomic>
#include <cassert>
#include <string>
#include <thread>
 
std::atomic<std::string*> ptr;
int data;
 
void producer()
{
    std::string* p = new std::string("Hello");
    data = 42;
    ptr.store(p, std::memory_order_release);
}
 
void consumer()
{
    std::string* p2;
    while (!(p2 = ptr.load(std::memory_order_acquire)))
        ;
    assert(*p2 == "Hello"); // never fires
    assert(data == 42); // never fires
}
 
int main()
{
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join(); t2.join();
}
