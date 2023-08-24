#include "Reduce_shader.h"

#include "../../compiler.h"
#include "Reduce_source.h"

namespace reduce_shader {

std::vector<uint32_t> generate_reduce_shader(
    Generate_reduce_shader_info generate_reduce_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_reduce_shader_info.dt);

  return shader_wizard::compile_shader("Reduce", reduce_source, options);
}

}  // namespace reduce_shader