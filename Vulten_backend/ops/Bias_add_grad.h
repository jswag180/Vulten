#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {

enum Channel_format {
  NHWC = 0,
  NCHW = 1,
};

class Bias_add_grad_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor input, Channel_format channel_format,
              Vulten_tensor output);

  Bias_add_grad_op(vulten_backend::Instance *inst);
  ~Bias_add_grad_op();
};
}  // namespace vulten_ops