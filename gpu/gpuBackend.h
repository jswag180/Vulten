#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <source_location>
#include <unordered_map>
#include <vector>

#define LOAD_SHADER_TO_VEC(v, n) \
  v = std::vector<uint32_t>((uint32_t*)n, (uint32_t*)(n + n##_len));

#define HANDLE_TO_GPUBACKEND_PTR(h) static_cast<gpuBackend*>(h)

#ifdef OP_TIMERS
#define SCOPE_TIMER(n) utills::ScopeTimer scopeTimer(n);
#else
#define SCOPE_TIMER(n)
#endif

struct MutexScopeLock {
  std::mutex* m_mutex;
  bool isLocked = false;

  MutexScopeLock() { m_mutex = nullptr; };

  MutexScopeLock(std::mutex* mutex) {
    m_mutex = mutex;
    m_mutex->lock();
    isLocked = true;
  };

  void setAndLockMutex(std::mutex* mutex) {
    m_mutex = mutex;
    m_mutex->lock();
    isLocked = true;
  }

  void unlock() {
    if (!isLocked) {
      m_mutex->unlock();
      isLocked = false;
    }
  }

  ~MutexScopeLock() {
    if (m_mutex != nullptr && isLocked) {
      m_mutex->unlock();
    }
  };
};

struct QueueProps {
  int family, queues;
  bool hasCompute, hasTransfer, hasGraphics;

  bool isFullyEnabled() { return hasCompute && hasTransfer && hasGraphics; }
};

struct devicePropertys {
  vk::PhysicalDeviceProperties physicalProperties;
  std::vector<std::string> deviceExtentions;
  vk::PhysicalDeviceMemoryProperties memProperties;
  std::vector<QueueProps> queueProperties;
};

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

class gpuBackend {
 private:
  static int numDevices;

  std::unordered_map<void*, std::shared_ptr<kp::TensorT<float>>> tensors;
  std::mutex buffers_mutex;

 public:
  static const int INFO = 0;
  static const int ERROR = 1;

  static std::vector<gpuBackend*> instances;
  static std::vector<devicePropertys> deviceProps;

  int device;
  kp::Manager* mngr;

  bool hasMemQueue = false;
  std::mutex memoryQueueMutex;
  int memoryQueue = -1;
  bool hasTransQueue = false;
  std::mutex transferQueueMutex;
  int transferQueue = -1;

  std::mutex mainQueueMutex;
  int mainQueue = -1;

  gpuBackend(int device);

  static void vultenLog(const int level, const char* mseg);
  static int listDevices();
  static std::vector<uint32_t> compileSource(const std::string& source);
  std::shared_ptr<kp::TensorT<float>>* addBuffer(uint64_t size);
  std::shared_ptr<kp::TensorT<float>>* getBuffer(void* tensorPtr);
  bool isDeviceBuffer(void* tensorPtr);
  void removeBuffer(void* tensorPtr);

  kp::Manager* getManager();

  ~gpuBackend();
};
