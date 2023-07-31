#include "ReluGrad_shader.h"

#include "../../compiler.h"
#include "ReluGrad_source.h"

std::vector<uint32_t> generate_reluGrad_shader(
    Generate_reluGrad_shader_info generate_reluGrad_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  options.AddMacroDefinition(
      "TYPE_0", vulten_ops::Data_type_to_str(generate_reluGrad_shader_info.dt));
  options.AddMacroDefinition(
      "TYPE_NUM_0", std::to_string(uint32_t(generate_reluGrad_shader_info.dt)));

  return shader_wizard::compile_shader("ReluGrad", reluGrad_source, options);
}