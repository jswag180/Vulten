#include "Basic_shader.h"

#include "../../compiler.h"
#include "Basic_source.h"

namespace basic_shader {

std::vector<uint32_t> generate_basic_shader(
    Generate_basic_shader_info generate_basic_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_basic_shader_info.dt);

  return shader_wizard::compile_shader("Basic", basic_source, options);
}

}  // namespace basic_shader