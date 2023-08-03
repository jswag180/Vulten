#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_transpose_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_transpose_shader(
    Generate_transpose_shader_info generate_transpose_shader_info);