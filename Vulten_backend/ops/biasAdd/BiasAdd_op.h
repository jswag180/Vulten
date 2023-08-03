#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class BiasAdd_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor bias,
              uint32_t bias_dim, Vulten_tensor output);

  BiasAdd_op(vulten_backend::Instance *inst);
  ~BiasAdd_op();
};
}  // namespace vulten_ops