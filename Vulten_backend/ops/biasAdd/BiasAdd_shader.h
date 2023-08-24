#pragma once

#include "../Vulten_backend_ops.h"

namespace bias_add_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Push_const {
  uint32_t bias_dim;
};

struct Generate_biasAdd_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_biasAdd_shader(
    Generate_biasAdd_shader_info generate_biasAdd_shader_info);

}  // namespace bias_add_shader