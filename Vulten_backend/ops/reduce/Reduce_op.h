#pragma once

#include "../Vulten_backend_ops.h"

#define OP_SUM 0
#define OP_MAX 1
#define OP_MIN 2

namespace vulten_ops {
class Reduce_op : Vulten_op {
 private:
  //

 public:
  //
  static std::string op_as_str(uint32_t op);
  void run_op(Data_type dt, Vulten_tensor input, std::vector<int32_t> &axis,
              Vulten_tensor output, uint32_t op);

  Reduce_op(vulten_backend::Instance *inst);
  ~Reduce_op();
};
}  // namespace vulten_ops