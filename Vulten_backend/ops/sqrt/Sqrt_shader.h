#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_sqrt_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_sqrt_shader(
    Generate_sqrt_shader_info generate_sqrt_shader_info);