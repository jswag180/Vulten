#include "Vulten_backend_ops.h"

#include <filesystem>
#include <fstream>

#include "shaderc/shaderc.hpp"
#include "shaders/headers/prelude/prelude.h.h"

namespace vulten_ops {

std::string Data_type_to_str(Data_type dt) {
  if (dt == Data_type::VULTEN_FLOAT) {
    return "float";
  } else if (dt == Data_type::VULTEN_FLOAT16) {
    return "float16_t";
  } else if (dt == Data_type::VULTEN_DOUBLE) {
    return "double";
  } else if (dt == Data_type::VULTEN_INT32) {
    return "int";
  } else if (dt == Data_type::VULTEN_UINT32) {
    return "uint";
  } else if (dt == Data_type::VULTEN_INT64) {
    return "int64_t";
  } else if (dt == Data_type::VULTEN_UINT64) {
    return "uint64_t";
  } else if (dt == Data_type::VULTEN_INT8) {
    return "int8_t";
  } else if (dt == Data_type::VULTEN_UINT8) {
    return "uint8_t";
  } else if (dt == Data_type::VULTEN_INT16) {
    return "int16_t";
  } else if (dt == Data_type::VULTEN_UINT16) {
    return "uint16_t";
  } else if (dt == Data_type::VULTEN_COMPLEX64) {
    return "cx_64";
  } else if (dt == Data_type::VULTEN_COMPLEX128) {
    return "cx_128";
  } else if (dt == Data_type::VULTEN_BOOL) {
    return "bool8";
  } else {
    throw std::runtime_error(
        "Error not a valid vulten_ops::DataType passed to "
        "Data_type_to_str(Data_type dt).");
  }
}

Vulten_tensor::Vulten_tensor(vulten_backend::Buffer* buffer_ptr,
                             int64_t num_dims, int64_t* dims_ptr)
    : buffer(buffer_ptr), num_dims(num_dims), dims(dims_ptr) {
  //
}

int64_t Vulten_tensor::get_total_elements() {
  int64_t total_elements = 1;
  for (int64_t i = 0; i < num_dims; i++) {
    total_elements *= dims[i];
  }
  return total_elements;
}

Vulten_tensor::~Vulten_tensor() {
  //
}

Vulten_pipeline::Vulten_pipeline(vulten_backend::Instance& instance,
                                 uint32_t num_buffers,
                                 const std::vector<uint32_t>& shader_source,
                                 vk::SpecializationInfo* spec_info,
                                 std::vector<vk::PushConstantRange> push_ranges)
    : inst(&instance) {
  auto_clean = true;
  vk::ShaderModuleCreateInfo shader_create_info(vk::ShaderModuleCreateFlags(),
                                                shader_source.size() * 4,
                                                shader_source.data());
  shader = inst->logical_dev.createShaderModule(shader_create_info);

  std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_binding =
      std::vector<vk::DescriptorSetLayoutBinding>(num_buffers);
  for (uint32_t i = 0; i < num_buffers; i++) {
    descriptor_set_layout_binding[i] = {i, vk::DescriptorType::eStorageBuffer,
                                        1, vk::ShaderStageFlagBits::eCompute};
  }
  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
      vk::DescriptorSetLayoutCreateFlags(), descriptor_set_layout_binding);
  descriptor_set_layout = inst->logical_dev.createDescriptorSetLayout(
      descriptor_set_layout_create_info);

  vk::PipelineLayoutCreateInfo pipeline_layout_create_info(
      vk::PipelineLayoutCreateFlags(), descriptor_set_layout, push_ranges);

  pipeline_layout =
      inst->logical_dev.createPipelineLayout(pipeline_layout_create_info);
  pipeline_cache =
      inst->logical_dev.createPipelineCache(vk::PipelineCacheCreateInfo());

  vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
      vk::PipelineShaderStageCreateFlags(),  // Flags
      vk::ShaderStageFlagBits::eCompute,     // Stage
      shader,                                // Shader Module
      "main");                               // Shader Entry Point
  if (spec_info != nullptr && spec_info->mapEntryCount > 0) {
    pipeline_shader_create_info.setPSpecializationInfo(spec_info);
  }

  vk::ComputePipelineCreateInfo compute_pipeline_create_info(
      vk::PipelineCreateFlags(),    // Flags
      pipeline_shader_create_info,  // Shader Create Info struct
      pipeline_layout);             // Pipeline Layout
  pipeline =
      inst->logical_dev
          .createComputePipeline(pipeline_cache, compute_pipeline_create_info)
          .value;
}

Vulten_pipeline::Vulten_pipeline() { auto_clean = false; }

Vulten_pipeline::~Vulten_pipeline() {
  if (!auto_clean) return;
  inst->logical_dev.destroyShaderModule(shader);
  inst->logical_dev.destroyDescriptorSetLayout(descriptor_set_layout);
  inst->logical_dev.destroyPipelineLayout(pipeline_layout);
  inst->logical_dev.destroyPipelineCache(pipeline_cache);
  inst->logical_dev.destroyPipeline(pipeline);
}

class Includer : shaderc::CompileOptions::IncluderInterface {
  shaderc_include_result* GetInclude(const char* requested_source,
                                     shaderc_include_type type,
                                     const char* requesting_source,
                                     size_t include_depth) {
    shaderc_include_result* include_result = new shaderc_include_result();

    if (std::string(requested_source) == "prelude.h") {
      include_result->source_name = "prelude.h";
      include_result->source_name_length = strlen(include_result->source_name);

      include_result->content = prelude_h;
      include_result->content_length = strlen(prelude_h);
    }

    return include_result;
  }

  void ReleaseInclude(shaderc_include_result* data) { delete data; }
};

std::vector<uint32_t> compile_shader(const char* name, const char* source,
                                     Data_type* type_chain,
                                     uint32_t type_chain_size) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  for (uint32_t i = 0; i < type_chain_size; i++) {
    options.AddMacroDefinition("TYPE_" + std::to_string(i),
                               Data_type_to_str(type_chain[i]));
    options.AddMacroDefinition("TYPE_NUM_" + std::to_string(i),
                               std::to_string(uint32_t(type_chain[i])));
  }

  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_2);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetIncluder(
      std::unique_ptr<shaderc::CompileOptions::IncluderInterface>(
          (shaderc::CompileOptions::IncluderInterface*)new Includer()));

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, shaderc_compute_shader, name, options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << "Shader compile error:\n";
    std::cerr << module.GetErrorMessage();
    exit(-1);
  }

  if (std::getenv("VULTEN_DUMP_SPV") != nullptr) {
    if (std::string(std::getenv("VULTEN_DUMP_SPV")) == "1") {
      std::filesystem::path cwd =
          std::filesystem::current_path() / (std::string(name) + ".spv");

      std::vector<uint32_t> result_vec =
          std::vector<uint32_t>{module.begin(), module.end()};

      std::ofstream file(cwd.string());
      file.write(reinterpret_cast<char*>(result_vec.data()),
                 sizeof(uint32_t) * result_vec.size());
      file.close();
    }
  }

  return {module.begin(), module.end()};
}

bool Vulten_op::is_pipeline_cached(std::string pipe_string) {
  if (pipelines.find(pipe_string) == pipelines.end()) return false;
  return true;
}

Vulten_op::Vulten_op(vulten_backend::Instance* inst) : inst(inst) {
  //
}

void Vulten_op::run_op() {
  //
}

Vulten_pipeline* Vulten_op::create_pipeline(
    std::string pipe_string, uint32_t num_buffers, const char* shader_source,
    Data_type* type_chain, uint32_t type_chain_size,
    vk::SpecializationInfo* spec_info,
    std::vector<vk::PushConstantRange> push_ranges) {
  auto compiled_shader = compile_shader(pipe_string.c_str(), shader_source,
                                        type_chain, type_chain_size);

  pipelines[pipe_string] = new Vulten_pipeline(
      *inst, num_buffers, compiled_shader, spec_info, push_ranges);
  return pipelines[pipe_string];
}

Vulten_op::~Vulten_op() {
  for (auto pipe : pipelines) {
    delete pipe.second;
  }
}

}  // namespace vulten_ops