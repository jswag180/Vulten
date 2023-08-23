#include "MultiFunc_shader.h"

#include "../../compiler.h"
#include "MultiFunc_source.h"

std::vector<uint32_t> generate_multiFunc_shader(
    Generate_multiFunc_shader_info generate_multiFunc_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_multiFunc_shader_info.dt);

  return shader_wizard::compile_shader("MultiFunc", multiFunc_source, options);
}