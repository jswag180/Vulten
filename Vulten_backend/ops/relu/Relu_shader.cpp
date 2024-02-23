#include "Relu_shader.h"

#include "../../compiler.h"
#include "Relu_source.h"

namespace relu_shader {

std::vector<uint32_t> generate_relu_shader(
    Generate_relu_shader_info generate_relu_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_relu_shader_info.dt);

  return shader_wizard::compile_shader("Relu", relu_source, options);
}

}  // namespace relu_shader