#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Exp_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output);

  Exp_op(vulten_backend::Instance *inst);
  ~Exp_op();
};
}  // namespace vulten_ops