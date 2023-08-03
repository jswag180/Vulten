#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_assign_add_sub_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_assign_add_sub_shader(
    Generate_assign_add_sub_shader_info generate_assign_add_sub_shader_info);