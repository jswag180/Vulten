#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_exp_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_exp_shader(
    Generate_exp_shader_info generate_exp_shader_info);