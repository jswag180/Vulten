#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Pow_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, uint32_t scalar, Vulten_tensor x, Vulten_tensor y,
              Vulten_tensor output);

  Pow_op(vulten_backend::Instance *inst);
  ~Pow_op();
};
}  // namespace vulten_ops