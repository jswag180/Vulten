#pragma once

#include "../Vulten_backend_ops.h"

#define ADD 0
#define SUB 1

namespace vulten_ops {
class Assign_add_sub_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor value, int op);

  Assign_add_sub_op(vulten_backend::Instance *inst);
  ~Assign_add_sub_op();
};
}  // namespace vulten_ops