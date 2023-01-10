#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
template <Data_type T>
class Relu_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Vulten_tensor input, Vulten_tensor output);

  Relu_op(vulten_backend::Instance *inst);
  ~Relu_op();
};
}  // namespace vulten_ops