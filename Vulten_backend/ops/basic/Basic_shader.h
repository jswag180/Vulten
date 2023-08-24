#pragma once

#include "../Vulten_backend_ops.h"

namespace basic_shader {

struct Spec_cons {
  uint32_t local_x;
  uint32_t op;
};

struct Push_const {
  uint32_t x_ranks;
  uint32_t y_ranks;
  uint32_t out_ranks;
};

struct Generate_basic_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_basic_shader(
    Generate_basic_shader_info generate_basic_shader_info);

}  // namespace basic_shader