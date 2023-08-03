#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Sqrt_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output);

  Sqrt_op(vulten_backend::Instance *inst);
  ~Sqrt_op();
};
}  // namespace vulten_ops