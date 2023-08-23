#include "BiasAdd_shader.h"

#include "../../compiler.h"
#include "BiasAdd_source.h"

std::vector<uint32_t> generate_biasAdd_shader(
    Generate_biasAdd_shader_info generate_biasAdd_shader_info) {
  shaderc::CompileOptions options = shader_wizard::get_compile_options();

  shader_wizard::add_type_define(options, 0, generate_biasAdd_shader_info.dt);

  return shader_wizard::compile_shader("BiasAdd", biasAdd_source, options);
}