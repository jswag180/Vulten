#pragma once

#include <stdint.h>

#include <shaderc/shaderc.hpp>
#include <vector>

namespace shader_wizard {

shaderc::CompileOptions get_compile_options();
std::vector<uint32_t> compile_shader(const char* name, const char* source,
                                     shaderc::CompileOptions);

}  // namespace shader_wizard