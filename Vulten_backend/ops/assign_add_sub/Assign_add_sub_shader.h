#pragma once

#include "../Vulten_backend_ops.h"

namespace assign_add_sub_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Generate_assign_add_sub_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_assign_add_sub_shader(
    Generate_assign_add_sub_shader_info generate_assign_add_sub_shader_info);

}  // namespace assign_add_sub_shader