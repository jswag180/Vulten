#include "BiasAdd_op.h"

#include <cmath>

#include "BiasAdd_shader.h"

namespace vulten_ops {
namespace bias_add {

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor bias, uint32_t bias_dim, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::BiasAdd_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  std::string pipe_string = "BiasAdd_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::BiasAdd_op pipeline " + pipe_string)

    bias_add_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(bias_add_shader::Spec_cons, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0,
         sizeof(bias_add_shader::Push_const)}};

    bias_add_shader::Generate_biasAdd_shader_info generate_biasAdd_shader_info{
        dt};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS,
                                        vk::DescriptorType::eStorageBuffer);
    vulten_pipeline = inst->create_pipeline(
        pipe_string, buffer_types,
        generate_biasAdd_shader(generate_biasAdd_shader_info), &spec_info,
        push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::BiasAdd_op pipeline " +
                     pipe_string)
  }

  vulten_backend::Descriptor_set_alloc descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS,
                                vulten_pipeline->descriptor_set_layout);

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);
  vk::DescriptorBufferInfo bias_buffer_info(bias.buffer->vk_buffer, 0,
                                            bias.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &input_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &bias_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 2, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  cmd_buff.begin(cmd_buff_begin_info);
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        vulten_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets

  bias_add_shader::Push_const pushes = {bias_dim};
  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0,
                         sizeof(bias_add_shader::Push_const), &pushes);

  uint32_t threads = std::ceil(
      float(input.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

  cmd_buff.dispatch(threads, 1, 1);
  cmd_buff.end();
  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());

  vk::SubmitInfo SubmitInfo(0,           // Num Wait Semaphores
                            nullptr,     // Wait Semaphores
                            nullptr,     // Pipeline Stage Flags
                            1,           // Num Command Buffers
                            &cmd_buff);  // List of command buffers
  queue_alloc.queue->vk_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buff);
}

}  // namespace bias_add
}  // namespace vulten_ops