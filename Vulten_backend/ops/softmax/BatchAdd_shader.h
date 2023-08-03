#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_batchAdd_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_batchAdd_shader(
    Generate_batchAdd_shader_info generate_batchAdd_shader_info);