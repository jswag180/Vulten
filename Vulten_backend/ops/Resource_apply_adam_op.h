#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
class Resource_apply_adam_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor var, Vulten_tensor m, Vulten_tensor v,
              Vulten_tensor beta1_power, Vulten_tensor beta2_power,
              Vulten_tensor lr, Vulten_tensor beta1, Vulten_tensor beta2,
              Vulten_tensor epsilon, Vulten_tensor grad, bool use_nesterov);

  Resource_apply_adam_op(vulten_backend::Instance *inst);
  ~Resource_apply_adam_op();
};
}  // namespace vulten_ops