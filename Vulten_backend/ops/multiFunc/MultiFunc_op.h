#pragma once

#include "../Vulten_backend_ops.h"

#define OP_SQRT 0
#define OP_EXP 1
#define OP_LOG 2
#define OP_SQUARE 3
#define OP_NEG 4

namespace vulten_ops {
namespace multiFunc {
static const int NUM_BUFFERS = 2;
static const int NUM_SETS = 1;

std::string op_as_str(uint32_t op);
vulten_backend::Vulten_pipeline *get_multiFunc_pipeline(
    vulten_backend::Instance *inst, Data_type dt);
void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor output, uint32_t op);
}  // namespace multiFunc
}  // namespace vulten_ops