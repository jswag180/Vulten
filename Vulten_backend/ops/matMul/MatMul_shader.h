#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_matMul_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_matMul_shader(
    Generate_matMul_shader_info generate_matMul_shader_info);