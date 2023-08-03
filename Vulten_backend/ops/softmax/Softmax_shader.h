#pragma once

#include "../Vulten_backend_ops.h"

struct Generate_softmax_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_softmax_shader(
    Generate_softmax_shader_info generate_softmax_shader_info);