#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
struct Mat_size {
  uint32_t x, y;
};

class MatMul_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type dt, Vulten_tensor a, bool trans_a, Mat_size mat_size_a,
              Vulten_tensor b, bool trans_b, Mat_size mat_size_b,
              Vulten_tensor output);

  MatMul_op(vulten_backend::Instance *inst);
  ~MatMul_op();
};
}  // namespace vulten_ops