#include "ReluGrad_op.h"

#include "shaders/headers/ReluGrad/ReluGrad.comp.h"

#define NUM_BUFFERS 3
#define NUM_SETS 1

namespace vulten_ops {

ReluGrad_op::ReluGrad_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::ReluGrad_op")
}

void ReluGrad_op::run_op(Data_type dt, Vulten_tensor gradients,
                         Vulten_tensor features, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::ReluGrad_op<" + Data_type_to_str(dt) +
                   ">")
  inst->main_queue_mutex.lock();

  std::string pipe_string = "ReluGrad_" + Data_type_to_str(dt);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::ReluGrad_op pipeline " + pipe_string)
    std::vector<Data_type> type_chain = {dt};
    vulten_pipeline = create_pipeline(pipe_string, NUM_BUFFERS, ReluGrad_comp,
                                      type_chain.data(), type_chain.size());
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::ReluGrad_op pipeline " +
                     pipe_string)
    vulten_pipeline = pipelines[pipe_string];
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
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
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
  inst->main_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(inst->cmd_pool, cmd_buff);
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

ReluGrad_op::~ReluGrad_op() {
  VULTEN_LOG_DEBUG("Freeing vulten_ops::ReluGrad_op")
}

}  // namespace vulten_ops