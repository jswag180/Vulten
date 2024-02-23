#pragma once

#include "../Vulten_backend_ops.h"

namespace resource_apply_adam_shader {

struct Spec_cons {
  uint32_t localX;
  bool nesterov;
};

struct Generate_resource_apply_adam_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_resource_apply_adam_shader(
    Generate_resource_apply_adam_shader_info
        generate_resource_apply_adam_shader_info);

}  // namespace resource_apply_adam_shader