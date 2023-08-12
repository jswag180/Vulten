#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Softmax_op : Vulten_op {
 private:
  //
 public:
  vulten_ops::Vulten_pipeline* get_multiFunc_pipeline(Data_type dt);
  vulten_ops::Vulten_pipeline* get_batchAdd_pipeline(Data_type dt);
  vulten_ops::Vulten_pipeline* get_softmax_pipeline(Data_type dt);

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output);

  Softmax_op(vulten_backend::Instance* inst);
  ~Softmax_op();
};
}  // namespace vulten_ops