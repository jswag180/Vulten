#pragma once

#include "../Vulten_backend_ops.h"

namespace mat_mul_shader {

struct Generate_matMul_shader_info {
  vulten_ops::Data_type dt;
};

std::vector<uint32_t> generate_matMul_shader(
    Generate_matMul_shader_info generate_matMul_shader_info);

}  // namespace mat_mul_shader