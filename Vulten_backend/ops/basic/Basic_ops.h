#pragma once

#include "../Vulten_backend_ops.h"

#define OP_MUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_DIV 3
#define OP_DIV_NO_NAN 4
#define OP_MAXIMUM 5
#define OP_MINIMUM 6
#define OP_DIV_REAL 7
#define OP_LOGICAL_AND 8
#define OP_LOGICAL_OR 9
#define OP_LESS 10
#define OP_LESS_EQUAL 11
#define OP_GREATER 12
#define OP_GREATER_EQUAL 13

namespace vulten_ops {
namespace basic {
static const int NUM_BUFFERS = 5;
static const int NUM_SETS = 1;

std::string op_as_str(uint32_t op);
void run_op(vulten_backend::Instance *inst, Data_type dt, uint32_t op,
            Vulten_tensor x, Vulten_tensor y, Vulten_tensor output);
}  // namespace basic
}  // namespace vulten_ops