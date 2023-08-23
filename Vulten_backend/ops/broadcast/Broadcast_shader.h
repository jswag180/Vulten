#pragma once

#include "../Vulten_backend_ops.h"

namespace broadcast_shader {

struct Spec_cons {
  uint32_t local_x;
};

struct Push_const {
  uint32_t in_ranks;
  uint32_t out_ranks;
};

struct Generate_broadcast_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_broadcast_shader(
    Generate_broadcast_shader_info generate_broadcast_shader_info);

}  // namespace broadcast_shader