#pragma once

#include "../Vulten_backend_ops.h"

namespace reluGrad_shader {

struct Generate_reluGrad_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_reluGrad_shader(
    Generate_reluGrad_shader_info generate_reluGrad_shader_info);

}  // namespace reluGrad_shader