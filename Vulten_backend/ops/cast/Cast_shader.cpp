#include "Cast_shader.h"

#include "../../compiler.h"
#include "Cast_source.h"

std::vector<uint32_t> generate_cast_shader(
    Generate_cast_shader_info generate_cast_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  options.AddMacroDefinition(
      "TYPE_0", vulten_ops::Data_type_to_str(generate_cast_shader_info.src));
  options.AddMacroDefinition(
      "TYPE_NUM_0", std::to_string(uint32_t(generate_cast_shader_info.src)));

  options.AddMacroDefinition(
      "TYPE_1", vulten_ops::Data_type_to_str(generate_cast_shader_info.dst));
  options.AddMacroDefinition(
      "TYPE_NUM_1", std::to_string(uint32_t(generate_cast_shader_info.dst)));

  return shader_wizard::compile_shader("Cast", cast_source, options);
}