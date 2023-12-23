#pragma once

#include "../Vulten_backend_ops.h"

namespace mat_mul_shader {

struct Spec_cons {
  uint32_t local_x;
  uint32_t local_y;
  uint32_t blockSizeX;
  uint32_t blockSizeY;
  uint32_t bkNum;
  VkBool32 transA;
  VkBool32 transB;
};

struct Push_const {
  uint32_t aDimsX, aDimsY;
  uint32_t bDimsX, bDimsY;
};

struct Generate_matMul_shader_info {
  vulten_ops::Data_type dt;
  bool unroll_bk;
};

std::vector<uint32_t> generate_matMul_shader(
    Generate_matMul_shader_info generate_matMul_shader_info);

}  // namespace mat_mul_shader