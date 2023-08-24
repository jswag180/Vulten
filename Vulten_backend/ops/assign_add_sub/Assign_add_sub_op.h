#pragma once

#include "../Vulten_backend_ops.h"

#define ADD 0
#define SUB 1

namespace vulten_ops {
namespace assign_add_sub {
static const int NUM_BUFFERS = 2;
static const int NUM_SETS = 1;

vulten_backend::Vulten_pipeline *get_assign_add_sub_pipeline(
    vulten_backend::Instance *inst, Data_type dt);
void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor value, int op);
}  // namespace assign_add_sub
}  // namespace vulten_ops