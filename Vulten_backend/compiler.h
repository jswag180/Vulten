#pragma once

#include <stdint.h>

#include <shaderc/shaderc.hpp>
#include <vector>

#include "Vulten_backend/ops/Vulten_backend_ops.h"

namespace shader_wizard {

shaderc::CompileOptions get_compile_options();
std::vector<uint32_t> compile_shader(const char* name, const char* source,
                                     shaderc::CompileOptions);
static inline void add_type_define(shaderc::CompileOptions& options,
                                   uint32_t num, vulten_ops::Data_type& dt) {
  options.AddMacroDefinition("TYPE_" + std::to_string(num),
                             vulten_ops::Data_type_to_str(dt));
  options.AddMacroDefinition("TYPE_NUM_" + std::to_string(num),
                             std::to_string(uint32_t(dt)));
}

}  // namespace shader_wizard