#include "Vulten_backend.h"

namespace vulten_backend {

Vulten_pipeline::Vulten_pipeline(vulten_backend::Instance* instance,
                                 std::vector<vk::DescriptorType>& buffer_types,
                                 const std::vector<uint32_t>& shader_source,
                                 vk::SpecializationInfo* spec_info,
                                 std::vector<vk::PushConstantRange> push_ranges)
    : inst(instance) {
  auto_clean = true;
  vk::ShaderModuleCreateInfo shader_create_info(vk::ShaderModuleCreateFlags(),
                                                shader_source.size() * 4,
                                                shader_source.data());
  shader = inst->logical_dev.createShaderModule(shader_create_info);

  std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_binding =
      std::vector<vk::DescriptorSetLayoutBinding>(buffer_types.size());
  for (uint32_t i = 0; i < buffer_types.size(); i++) {
    descriptor_set_layout_binding[i] = {i, buffer_types[i], 1,
                                        vk::ShaderStageFlagBits::eCompute};
  }
  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
      vk::DescriptorSetLayoutCreateFlags(), descriptor_set_layout_binding);
  descriptor_set_layout = inst->logical_dev.createDescriptorSetLayout(
      descriptor_set_layout_create_info);

  vk::PipelineLayoutCreateInfo pipeline_layout_create_info(
      vk::PipelineLayoutCreateFlags(), descriptor_set_layout, push_ranges);

  pipeline_layout =
      inst->logical_dev.createPipelineLayout(pipeline_layout_create_info);

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
  pipeline = inst->logical_dev
                 .createComputePipeline(inst->pipeline_cache,
                                        compute_pipeline_create_info)
                 .value;
}

Vulten_pipeline::Vulten_pipeline() { auto_clean = false; }

Vulten_pipeline::~Vulten_pipeline() {
  if (!auto_clean) return;
  inst->logical_dev.destroyShaderModule(shader);
  inst->logical_dev.destroyDescriptorSetLayout(descriptor_set_layout);
  inst->logical_dev.destroyPipelineLayout(pipeline_layout);
  inst->logical_dev.destroyPipeline(pipeline);
}

}  // namespace vulten_backend