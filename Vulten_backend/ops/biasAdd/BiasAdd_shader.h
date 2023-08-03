#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_biasAdd_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_biasAdd_shader(
    Generate_biasAdd_shader_info generate_biasAdd_shader_info);