#pragma once

#include "Vulten_backend_ops.h"

namespace vulten_ops {
template <Data_type SRC, Data_type DST>
class Cast_op : Vulten_op {
 private:
  //
 public:
  //

  void run_op(Vulten_tensor input, Vulten_tensor output);

  Cast_op(vulten_backend::Instance *inst);
  ~Cast_op();
};
}  // namespace vulten_ops