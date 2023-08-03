#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_resource_apply_adam_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_resource_apply_adam_shader(
    Generate_resource_apply_adam_shader_info
        generate_resource_apply_adam_shader_info);