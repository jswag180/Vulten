#pragma once

#include "../Vulten_backend_ops.h"

namespace cast_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Generate_cast_shader_info {
  vulten_ops::Data_type src;
  vulten_ops::Data_type dst;
};

std::vector<uint32_t> generate_cast_shader(
    Generate_cast_shader_info generate_cast_shader_info);
}  // namespace cast_shader