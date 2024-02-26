#pragma once

#include <stdint.h>

#include <string>
#include <vector>

namespace vulten_utills {

// clang-format off
inline std::vector<uint32_t> calculate_adj_strides(std::vector<int64_t> &
                                                    dims) {
  std::vector<uint32_t> adj_strides = std::vector<uint32_t>(dims.size() + 1, 1);

  for (int64_t i = 0; i < dims.size(); i++) {
    for (int64_t j = i; j < dims.size(); j++) {
      adj_strides[i] *= dims[j];
    }
  }

  return adj_strides;
}


inline std::vector<uint32_t> calculate_adj_strides(int64_t*
                                                    dims, uint64_t size) {
  std::vector<uint32_t> adj_strides = std::vector<uint32_t>(size + 1, 1);

  for (int64_t i = 0; i < size; i++) {
    for (int64_t j = i; j < size; j++) {
      adj_strides[i] *= dims[j];
    }
  }

  return adj_strides;
}
// clang-format on

static inline bool get_env_bool(const char* var) {
  auto env_var = std::getenv(var);
  if (env_var != nullptr) {
    if (std::string(env_var) == "true") {
      return true;
    }
  }
  return false;
}

};  // namespace vulten_utills