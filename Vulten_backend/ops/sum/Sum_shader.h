#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_sum_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_sum_shader(
    Generate_sum_shader_info generate_sum_shader_info);