#include "Transpose_shader.h"

#include "../../compiler.h"
#include "Transpose_source.h"

std::vector<uint32_t> generate_transpose_shader(
    Generate_transpose_shader_info generate_transpose_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_transpose_shader_info.dt);

  return shader_wizard::compile_shader("Transpose", transpose_source, options);
}