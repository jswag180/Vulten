#pragma once

#include <unordered_map>

#include "../Vulten_backend.h"

namespace vulten_ops {

enum Data_type {
  VULTEN_FLOAT = 1,
  VULTEN_FLOAT16 = 19,
  VULTEN_DOUBLE = 2,
  VULTEN_INT32 = 3,
  VULTEN_UINT32 = 22,
  VULTEN_INT64 = 9,
  VULTEN_UINT64 = 23,
  VULTEN_INT8 = 6,
  VULTEN_UINT8 = 4,
  VULTEN_INT16 = 5,
  VULTEN_UINT16 = 17,
  VULTEN_COMPLEX64 = 8,
  VULTEN_COMPLEX128 = 18,
  VULTEN_BOOL = 10,
};

#define VULTEN_DEFINE_BASIC_TYPES(op)                                     \
  op(VULTEN_FLOAT) op(VULTEN_FLOAT16) op(VULTEN_DOUBLE) op(VULTEN_INT32)  \
      op(VULTEN_UINT32) op(VULTEN_INT8) op(VULTEN_UINT8) op(VULTEN_INT64) \
          op(VULTEN_UINT64)

std::string Data_type_to_str(Data_type dt);

struct Vulten_tensor {
  vulten_backend::Buffer *buffer;
  int64_t *dims;
  int64_t num_dims;

  int64_t get_total_elements();

  Vulten_tensor(vulten_backend::Buffer *buffer_ptr, int64_t num_dims,
                int64_t *dims_ptr);
  Vulten_tensor() : buffer(nullptr), dims(nullptr), num_dims(0){};
  ~Vulten_tensor();
};

}  // namespace vulten_ops
