#include "Assign_add_sub_shader.h"

#include "../../compiler.h"
#include "Assign_add_sub_source.h"

namespace assign_add_sub_shader {

std::vector<uint32_t> generate_assign_add_sub_shader(
    Generate_assign_add_sub_shader_info generate_assign_add_sub_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0,
                                 generate_assign_add_sub_shader_info.dt);

  return shader_wizard::compile_shader("Assign_add_sub", assign_add_sub_source,
                                       options);
}

}  // namespace assign_add_sub_shader