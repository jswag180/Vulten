#include "Broadcast_shader.h"

#include "../../compiler.h"
#include "Broadcast_source.h"

namespace broadcast_shader {

std::vector<uint32_t> generate_broadcast_shader(
    Generate_broadcast_shader_info generate_broadcast_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_broadcast_shader_info.dt);

  return shader_wizard::compile_shader("Broadcast", broadcast_source, options);
}

}  // namespace broadcast_shader