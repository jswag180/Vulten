#include "MatMul_shader.h"

#include "../../compiler.h"
#include "MatMul_source.h"

namespace mat_mul_shader {

std::vector<uint32_t> generate_matMul_shader(
    Generate_matMul_shader_info generate_matMul_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_matMul_shader_info.dt);

  return shader_wizard::compile_shader("MatMul", matMul_source, options);
}

}  // namespace mat_mul_shader