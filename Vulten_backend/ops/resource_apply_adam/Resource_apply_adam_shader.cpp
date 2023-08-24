#include "Resource_apply_adam_shader.h"

#include "../../compiler.h"
#include "Resource_apply_adam_source.h"

namespace resource_apply_adam_shader {

std::vector<uint32_t> generate_resource_apply_adam_shader(
    Generate_resource_apply_adam_shader_info
        generate_resource_apply_adam_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0,
                                 generate_resource_apply_adam_shader_info.dt);

  return shader_wizard::compile_shader("Resource_apply_adam",
                                       resource_apply_adam_source, options);
}

}  // namespace resource_apply_adam_shader