#pragma once

#include <chrono>
#include <iostream>
#include <string>

#ifdef OP_TIMERS
#define SCOPE_TIMER(n) utills::ScopeTimer scopeTimer(n);
#else
#define SCOPE_TIMER(n)
#endif

namespace utills {
class ScopeTimer {
 private:
  std::string name_;
  std::chrono::_V2::system_clock::time_point start;
  std::chrono::_V2::system_clock::time_point stamp;

 public:
  ScopeTimer(std::string name) {
    name_ = name;
    start = std::chrono::high_resolution_clock::now();
    stamp = start;
  };

  void segment(int line) {
    auto newStamp = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(newStamp - stamp);
    std::cout << "Timmer segment: " << name_ << " At line: " << line
              << " Took: " << duration.count()
              << " microseconds from last segment/start\n";
    stamp = newStamp;
  };

  ~ScopeTimer() {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Timmer: " << name_ << " Took: " << duration.count()
              << " microseconds\n";
  };
};
}  // namespace utills