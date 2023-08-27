#include "ReluGrad_op.h"

#include "ReluGrad_shader.h"

namespace vulten_ops {
namespace reluGrad {

void run_op(vulten_backend::Instance *inst, Data_type dt,
            Vulten_tensor gradients, Vulten_tensor features,
            Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::ReluGrad_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  std::string pipe_string = "ReluGrad_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::ReluGrad_op pipeline " + pipe_string)
    reluGrad_shader::Generate_reluGrad_shader_info
        generate_reluGrad_shader_info{dt};
    vulten_pipeline =
        inst->create_pipeline(pipe_string, NUM_BUFFERS,
                              reluGrad_shader::generate_reluGrad_shader(
                                  generate_reluGrad_shader_info));
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::ReluGrad_op pipeline " +
                     pipe_string)
  }

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer, NUM_BUFFERS);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, NUM_SETS,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, NUM_SETS, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo gradients_buffer_info(gradients.buffer->vk_buffer, 0,
                                                 gradients.buffer->buffer_size);
  vk::DescriptorBufferInfo features_buffer_info(features.buffer->vk_buffer, 0,
                                                features.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &gradients_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &features_buffer_info},
      {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &output_buffer_info},
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
      {descriptor_set},                  // List of descriptor sets
      {});                               // Dynamic offsets
  cmd_buff.dispatch(uint32_t(gradients.get_total_elements()), 1, 1);
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
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
}

}  // namespace reluGrad
}  // namespace vulten_ops