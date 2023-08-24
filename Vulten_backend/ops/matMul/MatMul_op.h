#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace mat_mul {
static const int NUM_BUFFERS = 3;
static const int NUM_SETS = 1;
struct Mat_size {
  uint32_t x, y;
};

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor a,
            bool trans_a, Mat_size mat_size_a, Vulten_tensor b, bool trans_b,
            Mat_size mat_size_b, Vulten_tensor output);
}  // namespace mat_mul
}  // namespace vulten_ops