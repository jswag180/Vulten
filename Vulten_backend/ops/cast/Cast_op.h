#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace cast {
static const int NUM_BUFFERS = 2;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type src, Data_type dst,
            Vulten_tensor input, Vulten_tensor output);
}  // namespace cast
}  // namespace vulten_ops