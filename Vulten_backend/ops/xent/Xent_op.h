#pragma once

#include "../Vulten_backend_ops.h"

#define OP_LOSS 0
#define OP_GRAD 1

namespace vulten_ops {
class Xent_op : Vulten_op {
 private:
  //
 public:
  //
  void run_op(Data_type dt, Data_type dt_labels, Vulten_tensor scratch,
              Vulten_tensor backprop, Vulten_tensor labels,
              Vulten_tensor output, uint32_t op);

  Xent_op(vulten_backend::Instance *inst);
  ~Xent_op();
};
}  // namespace vulten_ops