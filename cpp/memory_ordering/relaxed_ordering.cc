// Typical use for relaxed memory ordering is incrementing counters, such as the reference counters of std::shared_ptr,
// since this only requires atomicity, but not ordering or synchronization
// (note that decrementing the std::shared_ptr counters requires acquire-release synchronization with the destructor).

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
 
std::atomic<int> cnt = {0};
 
void f()
{
    for (int n = 0; n < 1000; ++n)
        cnt.fetch_add(1, std::memory_order_relaxed);
}
 
int main()
{
    std::vector<std::thread> v;
    for (int n = 0; n < 10; ++n)
        v.emplace_back(f);
    for (auto& t : v)
        t.join();
    std::cout << "Final counter value is " << cnt << '\n';
}
