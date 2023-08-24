#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace reluGrad {
static const int NUM_BUFFERS = 3;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type dt,
            Vulten_tensor gradients, Vulten_tensor features,
            Vulten_tensor output);
}  // namespace reluGrad
}  // namespace vulten_ops