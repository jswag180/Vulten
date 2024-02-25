#include "compiler.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Vulten_backend/Vulten_utills.h"
#include "prelude.h"

namespace shader_wizard {

class Includer : shaderc::CompileOptions::IncluderInterface {
  shaderc_include_result* GetInclude(const char* requested_source,
                                     shaderc_include_type type,
                                     const char* requesting_source,
                                     size_t include_depth) {
    shaderc_include_result* include_result = new shaderc_include_result();

    if (std::string(requested_source) == "prelude.h") {
      include_result->source_name = "prelude.h";
      include_result->source_name_length = strlen(include_result->source_name);

      include_result->content = prelude_header;
      include_result->content_length = strlen(prelude_header);
    } else {
      std::cerr << "Unknown include: " << std::string(requested_source) << "\n";
      exit(-1);
    }

    return include_result;
  }

  void ReleaseInclude(shaderc_include_result* data) { delete data; }
};

shaderc::CompileOptions get_compile_options() {
  shaderc::CompileOptions options;

  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_2);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetIncluder(
      std::unique_ptr<shaderc::CompileOptions::IncluderInterface>(
          (shaderc::CompileOptions::IncluderInterface*)new Includer()));

  return options;
}

std::vector<uint32_t> compile_shader(const char* name, const char* source,
                                     shaderc::CompileOptions options) {
  shaderc::Compiler compiler;

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, shaderc_compute_shader, name, options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << "Shader compile error:\n";
    std::cerr << module.GetErrorMessage();
    exit(-1);
  }

  if (vulten_utills::get_env_bool("VULTEN_DUMP_SPV")) {
    std::filesystem::path cwd =
        std::filesystem::current_path() / (std::string(name) + ".spv");

    std::vector<uint32_t> result_vec =
        std::vector<uint32_t>{module.begin(), module.end()};

    std::ofstream file(cwd.string());
    file.write(reinterpret_cast<char*>(result_vec.data()),
               sizeof(uint32_t) * result_vec.size());
    file.close();
  }

  return {module.begin(), module.end()};
}

}  // namespace shader_wizard
