#pragma once

#include "../Vulten_backend_ops.h"

namespace multiFunc_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Push_const {
  uint32_t op;
};

struct Generate_multiFunc_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_multiFunc_shader(
    Generate_multiFunc_shader_info generate_multiFunc_shader_info);

}  // namespace multiFunc_shader