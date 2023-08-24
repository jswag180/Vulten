#pragma once

#include "../Vulten_backend_ops.h"

namespace transpose_shader {

struct Push_const {
  uint32_t hight;
  uint32_t width;
};

struct Generate_transpose_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_transpose_shader(
    Generate_transpose_shader_info generate_transpose_shader_info);

}  // namespace transpose_shader