#pragma once

#include "../Vulten_backend_ops.h"

#define OP_MUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_DIV 3
#define OP_DIV_NO_NAN 4

namespace vulten_ops {
namespace basic {
static const int NUM_BUFFERS = 5;
static const int NUM_SETS = 1;

std::string op_as_str(uint32_t op);
void run_op(vulten_backend::Instance *inst, Data_type dt, uint32_t op,
            Vulten_tensor x, Vulten_tensor y, Vulten_tensor output);
}  // namespace basic
}  // namespace vulten_ops