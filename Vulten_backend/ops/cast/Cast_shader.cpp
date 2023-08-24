#include "Cast_shader.h"

#include "../../compiler.h"
#include "Cast_source.h"

namespace cast_shader {

std::vector<uint32_t> generate_cast_shader(
    Generate_cast_shader_info generate_cast_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_cast_shader_info.src);
  shader_wizard::add_type_define(options, 1, generate_cast_shader_info.dst);

  return shader_wizard::compile_shader("Cast", cast_source, options);
}

}  // namespace cast_shader