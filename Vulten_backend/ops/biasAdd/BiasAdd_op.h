#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace bias_add {
static const int NUM_BUFFERS = 3;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor bias, uint32_t bias_dim, Vulten_tensor output);
}  // namespace bias_add
}  // namespace vulten_ops