#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
class Relu_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Vulten_tensor input, Vulten_tensor output);

  Relu_op(vulten_backend::Instance &inst, Data_type dt);
  ~Relu_op();
};
}  // namespace vulten_ops