#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
template <Data_type T>
class Assign_add_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Vulten_tensor input, Vulten_tensor value);

  Assign_add_op(vulten_backend::Instance *inst);
  ~Assign_add_op();
};
}  // namespace vulten_ops