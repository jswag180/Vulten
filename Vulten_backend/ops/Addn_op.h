#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
class Addn_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, std::vector<Vulten_tensor> &inputs,
              Vulten_tensor output);

  Addn_op(vulten_backend::Instance *inst);
  ~Addn_op();
};
}  // namespace vulten_ops