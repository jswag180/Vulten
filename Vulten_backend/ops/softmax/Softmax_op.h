#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace softmax {
static const int NUM_BUFFERS = 3;
static const int NUM_SETS = 3;

vulten_backend::Vulten_pipeline *get_batchAdd_pipeline(
    vulten_backend::Instance *inst, Data_type dt);
vulten_backend::Vulten_pipeline *get_softmax_pipeline(
    vulten_backend::Instance *inst, Data_type dt);

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor output);
}  // namespace softmax
}  // namespace vulten_ops