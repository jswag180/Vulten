#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_basic_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_basic_shader(
    Generate_basic_shader_info generate_basic_shader_info);