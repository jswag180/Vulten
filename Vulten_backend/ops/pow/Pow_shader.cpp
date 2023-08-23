#include "Pow_shader.h"

#include "../../compiler.h"
#include "Pow_source.h"

std::vector<uint32_t> generate_pow_shader(
    Generate_pow_shader_info generate_pow_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_pow_shader_info.dt);

  return shader_wizard::compile_shader("Pow", pow_source, options);
}