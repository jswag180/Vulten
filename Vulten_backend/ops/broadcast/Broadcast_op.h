#pragma once

#include "../Vulten_backend_ops.h"

namespace vulten_ops {
class Broadcast_op : Vulten_op {
 private:
  //
 public:
  static const std::string op_name;

  Vulten_pipeline* get_broadcast_pipeline(std::string pipe_string,
                                          Data_type dt);

  void run_op(Data_type dt, Vulten_tensor input, Vulten_tensor output);

  Broadcast_op(vulten_backend::Instance* inst);
  ~Broadcast_op();
};
}  // namespace vulten_ops