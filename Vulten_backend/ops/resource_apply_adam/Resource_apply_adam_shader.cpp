#include "Resource_apply_adam_shader.h"

#include "../../compiler.h"
#include "Resource_apply_adam_source.h"

std::vector<uint32_t> generate_resource_apply_adam_shader(
    Generate_resource_apply_adam_shader_info
        generate_resource_apply_adam_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  options.AddMacroDefinition("TYPE_0",
                             vulten_ops::Data_type_to_str(
                                 generate_resource_apply_adam_shader_info.dt));
  options.AddMacroDefinition(
      "TYPE_NUM_0",
      std::to_string(uint32_t(generate_resource_apply_adam_shader_info.dt)));

  return shader_wizard::compile_shader("Resource_apply_adam",
                                       resource_apply_adam_source, options);
}