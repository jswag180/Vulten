#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_pow_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_pow_shader(
    Generate_pow_shader_info generate_pow_shader_info);