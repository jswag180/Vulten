#include "BatchAdd_shader.h"

#include "../../compiler.h"
#include "BatchAdd_source.h"

std::vector<uint32_t> generate_batchAdd_shader(
    Generate_batchAdd_shader_info generate_batchAdd_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_batchAdd_shader_info.dt);

  return shader_wizard::compile_shader("BatchAdd", batchAdd_source, options);
}