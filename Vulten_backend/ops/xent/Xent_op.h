#pragma once

#include "../Vulten_backend_ops.h"

#define OP_LOSS 0
#define OP_GRAD 1

namespace vulten_ops {
namespace xent {
static const int NUM_BUFFERS = 4;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type dt, Data_type dt_labels,
            Vulten_tensor scratch, Vulten_tensor backprop, Vulten_tensor labels,
            Vulten_tensor output, uint32_t op);
}  // namespace xent
}  // namespace vulten_ops