#include "Softmax_shader.h"

#include "../../compiler.h"
#include "Softmax_source.h"

std::vector<uint32_t> generate_softmax_shader(
    Generate_softmax_shader_info generate_softmax_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  options.AddMacroDefinition(
      "TYPE_0", vulten_ops::Data_type_to_str(generate_softmax_shader_info.dt));
  options.AddMacroDefinition(
      "TYPE_NUM_0", std::to_string(uint32_t(generate_softmax_shader_info.dt)));

  return shader_wizard::compile_shader("Softmax", softmax_source, options);
}