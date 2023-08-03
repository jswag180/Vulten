#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Cast_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Data_type src, Data_type dst, Vulten_tensor input,
              Vulten_tensor output);

  Cast_op(vulten_backend::Instance *inst);
  ~Cast_op();
};
}  // namespace vulten_ops