#pragma once

#include "../Vulten_backend_ops.h"

#define OP_SQRT 0
#define OP_EXP 1
#define OP_LOG 2
#define OP_SQUARE 3

namespace vulten_ops {
class MultiFunc_op : Vulten_op {
 private:
  //
 public:
  //
  static std::string op_as_str(uint32_t op);
  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output,
              uint32_t op);

  MultiFunc_op(vulten_backend::Instance *inst);
  ~MultiFunc_op();
};
}  // namespace vulten_ops