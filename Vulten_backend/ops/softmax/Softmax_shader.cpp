#include "Softmax_shader.h"

#include "../../compiler.h"
#include "Softmax_source.h"

namespace softmax_shader {

std::vector<uint32_t> generate_softmax_shader(
    Generate_softmax_shader_info generate_softmax_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_softmax_shader_info.dt);

  return shader_wizard::compile_shader("Softmax", softmax_source, options);
}

}  // namespace softmax_shader