#pragma once

#include "Vulten_backend_ops.h"

#define OP_MUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_DIV 3
#define OP_DIV_NO_NAN 4

namespace vulten_ops {

std::string op_as_str(uint32_t op);

template <Data_type T>
class Basic_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(uint32_t op, Vulten_tensor x, Vulten_tensor y,
              Vulten_tensor output);

  Basic_op(vulten_backend::Instance *inst);
  ~Basic_op();
};
}  // namespace vulten_ops