#pragma once
#include <iostream>
#include <mutex>
#include <sstream>

#define BASEFILE(x)                                              \
  std::string_view(x).substr(std::string_view(x).rfind("/") + 1, \
                             std::string_view(x).rfind('.'))

#if VULTEN_LOG_LEVEL >= 1
#define VULTEN_LOG_INFO(M)                                                \
  VULTEN_LOG << "Vulten [INFO][" << BASEFILE(__FILE__) << ":" << __LINE__ \
             << "]: " << M << "\n";
#else
#define VULTEN_LOG_INFO(M)
#endif

#if VULTEN_LOG_LEVEL >= 2
#define VULTEN_LOG_DEBUG(M)                                                \
  VULTEN_LOG << "Vulten [DEBUG][" << BASEFILE(__FILE__) << ":" << __LINE__ \
             << "]: " << M << "\n";
#else
#define VULTEN_LOG_DEBUG(M)
#endif

#define VULTEN_LOG_ERROR(M)                                                \
  VULTEN_LOG << "Vulten [ERROR][" << BASEFILE(__FILE__) << ":" << __LINE__ \
             << "]: " << M << "\n";

class Vulten_logger {
 public:
  template <typename T>
  static void log(T& message) {
    mutex.lock();
    std::cout << message.str();
    message.flush();
    mutex.unlock();
  }

 private:
  static std::mutex mutex;
};
static Vulten_logger VULTEN_LOG;

struct Vulten_logger_buffer {
  std::stringstream ss;

  Vulten_logger_buffer() = default;
  Vulten_logger_buffer(const Vulten_logger_buffer&) = delete;
  Vulten_logger_buffer& operator=(const Vulten_logger_buffer&) = delete;
  Vulten_logger_buffer& operator=(Vulten_logger_buffer&&) = delete;
  Vulten_logger_buffer(Vulten_logger_buffer&& buf) : ss(std::move(buf.ss)) {}
  template <typename T>
  Vulten_logger_buffer& operator<<(T&& message) {
    ss << std::forward<T>(message);
    return *this;
  }

  ~Vulten_logger_buffer() { VULTEN_LOG.log(ss); }
};

template <typename T>
Vulten_logger_buffer operator<<(Vulten_logger& simpleLogger, T&& message) {
  Vulten_logger_buffer buf;
  buf.ss << std::forward<T>(message);
  return buf;
}