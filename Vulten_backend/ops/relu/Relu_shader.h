#pragma once

#include "../Vulten_backend_ops.h"
namespace relu_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Generate_relu_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_relu_shader(
    Generate_relu_shader_info generate_relu_shader_info);

}  // namespace relu_shader