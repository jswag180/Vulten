#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
namespace addn {
void run_op(vulten_backend::Instance *inst, Data_type dt,
            std::vector<Vulten_tensor> &inputs, Vulten_tensor output);
}
}  // namespace vulten_ops