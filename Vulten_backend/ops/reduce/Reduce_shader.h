#pragma once

#include "../Vulten_backend_ops.h"

namespace reduce_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Push_const {
  uint32_t axi_size;
  uint32_t adj_stride;
  uint32_t adj_stride_adv;
  uint32_t op;
};

struct Generate_reduce_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_reduce_shader(
    Generate_reduce_shader_info generate_reduce_shader_info);

}  // namespace reduce_shader