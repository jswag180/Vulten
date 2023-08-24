#pragma once

#include "../Vulten_backend_ops.h"

namespace xent_shader {
struct Spec_cons {
  uint32_t localX;
};

struct Push_const {
  uint32_t numLogits;
  uint32_t op;
};

struct Generate_xent_shader_info {
  vulten_ops::Data_type dt;
  vulten_ops::Data_type dt_labels;
};

std::vector<uint32_t> generate_xent_shader(
    Generate_xent_shader_info generate_xent_shader_info);

}  // namespace xent_shader