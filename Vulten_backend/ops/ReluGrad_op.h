#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
class ReluGrad_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor gradients, Vulten_tensor features,
              Vulten_tensor output);

  ReluGrad_op(vulten_backend::Instance *inst);
  ~ReluGrad_op();
};
}  // namespace vulten_ops