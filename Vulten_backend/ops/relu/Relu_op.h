#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Relu_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output);

  Relu_op(vulten_backend::Instance *inst);
  ~Relu_op();
};
}  // namespace vulten_ops