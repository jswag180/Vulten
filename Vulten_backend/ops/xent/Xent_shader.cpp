#include "Xent_shader.h"

#include "../../compiler.h"
#include "Xent_source.h"

std::vector<uint32_t> generate_xent_shader(
    Generate_xent_shader_info generate_xent_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  options.AddMacroDefinition(
      "TYPE_0", vulten_ops::Data_type_to_str(generate_xent_shader_info.dt));
  options.AddMacroDefinition(
      "TYPE_NUM_0", std::to_string(uint32_t(generate_xent_shader_info.dt)));

  options.AddMacroDefinition(
      "TYPE_1",
      vulten_ops::Data_type_to_str(generate_xent_shader_info.dt_labels));
  options.AddMacroDefinition(
      "TYPE_NUM_1",
      std::to_string(uint32_t(generate_xent_shader_info.dt_labels)));

  return shader_wizard::compile_shader("Xent", xent_source, options);
}