#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
class Sum_op : Vulten_op {
 private:
  //

 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, std::vector<int32_t> &axis,
              Vulten_tensor output);

  Sum_op(vulten_backend::Instance *inst);
  ~Sum_op();
};
}  // namespace vulten_ops