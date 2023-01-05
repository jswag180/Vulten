#include "Relu_op.h"

#include "shaders/headers/Relu/Relu.h"

namespace vulten_ops {

Relu_op::Relu_op(vulten_backend::Instance *inst, Data_type dt)
    : Vulten_op(inst, dt) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Relu_op")
}

void Relu_op::run_op(Vulten_tensor input, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Relu_op")
  inst->main_queue_mutex.lock();

  std::string pipe_string = "Relu_" + Data_type_to_str(data_type);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Relu_op pipeline " + pipe_string)
    vulten_pipeline = create_pipeline(pipe_string, 2, shader::Relu_float);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Relu_op pipeline " + pipe_string)
    vulten_pipeline = pipelines[pipe_string];
  }

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {vulten_pipeline->descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &input_buffer_info},
      {vulten_pipeline->descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &output_buffer_info},
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
      vk::PipelineBindPoint::eCompute,    // Bind point
      vulten_pipeline->pipeline_layout,   // Pipeline Layout
      0,                                  // First descriptor set
      {vulten_pipeline->descriptor_set},  // List of descriptor sets
      {});                                // Dynamic offsets
  cmd_buff.dispatch(uint32_t(input.get_total_elements()), 1, 1);
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
  inst->main_queue_mutex.unlock();
}

Relu_op::~Relu_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::Relu_op") }

}  // namespace vulten_ops