#pragma once

#include "../Vulten_backend_ops.h"

namespace batchAdd_shader {

struct Spec_cons {
  uint32_t localX;
};

struct Generate_batchAdd_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_batchAdd_shader(
    Generate_batchAdd_shader_info generate_batchAdd_shader_info);

}  // namespace batchAdd_shader