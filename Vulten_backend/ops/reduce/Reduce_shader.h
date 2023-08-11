#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_reduce_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_reduce_shader(
    Generate_reduce_shader_info generate_reduce_shader_info);