#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace broadcast {
static const int NUM_BUFFERS = 4;
static const int NUM_SETS = 1;

vulten_backend::Vulten_pipeline* get_broadcast_pipeline(
    vulten_backend::Instance* inst, Data_type dt);
void run_op(vulten_backend::Instance* inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor output);
}  // namespace broadcast
}  // namespace vulten_ops