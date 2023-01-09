#include "Vulten_backend_ops.h"

namespace vulten_ops {

std::string Data_type_to_str(Data_type dt) {
  if (dt == Data_type::VULTEN_FLOAT) {
    return "float";
  } else if (dt == Data_type::VULTEN_FLOAT16) {
    return "float16";
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
  } else {
    throw std::runtime_error(
        "Error not a valid vulten_ops::DataType passed to "
        "Data_type_to_str(Data_type dt).");
  }
}

Vulten_tensor::Vulten_tensor(vulten_backend::Buffer *buffer_ptr,
                             int64_t num_dims, int64_t *dims_ptr)
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

Vulten_pipeline::Vulten_pipeline(vulten_backend::Instance &instance,
                                 uint32_t num_buffers,
                                 const std::vector<uint32_t> &shader_source,
                                 vk::SpecializationInfo spec_info)
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
      vk::PipelineLayoutCreateFlags(), descriptor_set_layout);
  pipeline_layout =
      inst->logical_dev.createPipelineLayout(pipeline_layout_create_info);
  pipeline_cache =
      inst->logical_dev.createPipelineCache(vk::PipelineCacheCreateInfo());

  vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
      vk::PipelineShaderStageCreateFlags(),  // Flags
      vk::ShaderStageFlagBits::eCompute,     // Stage
      shader,                                // Shader Module
      "main");                               // Shader Entry Point
  if (spec_info.mapEntryCount > 0) {
    pipeline_shader_create_info.setPSpecializationInfo(&spec_info);
  }

  vk::ComputePipelineCreateInfo compute_pipeline_create_info(
      vk::PipelineCreateFlags(),    // Flags
      pipeline_shader_create_info,  // Shader Create Info struct
      pipeline_layout);             // Pipeline Layout
  pipeline =
      inst->logical_dev
          .createComputePipeline(pipeline_cache, compute_pipeline_create_info)
          .value;

  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer, num_buffers);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, 1, &descriptor_set_layout);
  const std::vector<vk::DescriptorSet> descriptor_sets =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info);
  descriptor_set = descriptor_sets.front();
}

Vulten_pipeline::Vulten_pipeline() { auto_clean = false; }

Vulten_pipeline::~Vulten_pipeline() {
  if (!auto_clean) return;
  inst->logical_dev.destroyShaderModule(shader);
  inst->logical_dev.destroyDescriptorSetLayout(descriptor_set_layout);
  inst->logical_dev.destroyPipelineLayout(pipeline_layout);
  inst->logical_dev.destroyPipelineCache(pipeline_cache);
  inst->logical_dev.destroyPipeline(pipeline);
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
}

bool Vulten_op::is_pipeline_cached(std::string pipe_string) {
  if (pipelines.find(pipe_string) == pipelines.end()) return false;
  return true;
}

Vulten_op::Vulten_op(vulten_backend::Instance *inst)
    : inst(inst) {
  //
}

void Vulten_op::run_op() {
  //
}

Vulten_pipeline *Vulten_op::create_pipeline(
    std::string pipe_string, uint32_t num_buffers,
    const std::vector<uint32_t> &shader_source,
    vk::SpecializationInfo spec_info) {
  pipelines[pipe_string] =
      new Vulten_pipeline(*inst, num_buffers, shader_source, spec_info);
  return pipelines[pipe_string];
}

Vulten_op::~Vulten_op() {
  for (auto pipe : pipelines) {
    delete pipe.second;
  }
}

}  // namespace vulten_ops