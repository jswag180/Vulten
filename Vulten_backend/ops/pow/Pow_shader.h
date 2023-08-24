#pragma once

#include "../Vulten_backend_ops.h"

namespace pow_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Push_const {
  uint32_t scalar;
};

struct Generate_pow_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_pow_shader(
    Generate_pow_shader_info generate_pow_shader_info);

}  // namespace pow_shader