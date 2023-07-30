#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_cast_shader_info {
  vulten_ops::Data_type src;
  vulten_ops::Data_type dst;
};

std::vector<uint32_t> generate_cast_shader(
    Generate_cast_shader_info generate_cast_shader_info);