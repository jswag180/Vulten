#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace resource_apply_adam {
static const int NUM_BUFFERS = 10;
static const int NUM_SETS = 1;

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor var,
            Vulten_tensor m, Vulten_tensor v, Vulten_tensor beta1_power,
            Vulten_tensor beta2_power, Vulten_tensor lr, Vulten_tensor beta1,
            Vulten_tensor beta2, Vulten_tensor epsilon, Vulten_tensor grad,
            bool use_nesterov);
}  // namespace resource_apply_adam
}  // namespace vulten_ops