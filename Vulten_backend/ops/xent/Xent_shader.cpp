#include "Xent_shader.h"

#include "../../compiler.h"
#include "Xent_source.h"

std::vector<uint32_t> generate_xent_shader(
    Generate_xent_shader_info generate_xent_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_xent_shader_info.dt);
  shader_wizard::add_type_define(options, 1,
                                 generate_xent_shader_info.dt_labels);

  return shader_wizard::compile_shader("Xent", xent_source, options);
}