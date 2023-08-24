#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace pow {
static const int NUM_BUFFERS = 3;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type dt, uint32_t scalar,
            Vulten_tensor x, Vulten_tensor y, Vulten_tensor output);
}  // namespace pow
}  // namespace vulten_ops