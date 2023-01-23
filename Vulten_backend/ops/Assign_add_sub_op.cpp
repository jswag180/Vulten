#include "Assign_add_sub_op.h"

#include "shaders/headers/AddSubInPlace/AddSubInPlace.h"

namespace vulten_ops {

#define DEFINE_ASSIGN_ADD(X) template class Assign_add_sub_op<X>;

VULTEN_DEFINE_BASIC_TYPES(DEFINE_ASSIGN_ADD)

template <Data_type T>
Assign_add_sub_op<T>::Assign_add_sub_op(vulten_backend::Instance *inst)
    : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Assign_add_sub_op<" +
                   Data_type_to_str(T) + ">")
}

template <Data_type T>
void Assign_add_sub_op<T>::run_op(Vulten_tensor input, Vulten_tensor value,
                                  int op) {
  if (op == ADD) {
    VULTEN_LOG_DEBUG("Running vulten_ops::Assign_add_op<" +
                     Data_type_to_str(T) + ">")
  } else if (op == SUB) {
    VULTEN_LOG_DEBUG("Running vulten_ops::Assign_sub_op<" +
                     Data_type_to_str(T) + ">")
  }
  inst->main_queue_mutex.lock();

  std::string pipe_string = "Assign_add_sub_" + Data_type_to_str(T);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Assign_add_sub_op pipeline " +
                     pipe_string)

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t)}};

    if (T == VULTEN_FLOAT) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_float, nullptr,
                          push_const_ranges);
    } else if (T == VULTEN_FLOAT16) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_float16_t,
                          nullptr, push_const_ranges);
    } else if (T == VULTEN_DOUBLE) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_double, nullptr,
                          push_const_ranges);
    } else if (T == VULTEN_INT32) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_int, nullptr,
                          push_const_ranges);
    } else if (T == VULTEN_UINT32) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_uint, nullptr,
                          push_const_ranges);
    } else if (T == VULTEN_INT8) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_int8_t, nullptr,
                          push_const_ranges);
    } else if (T == VULTEN_UINT8) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_uint8_t,
                          nullptr, push_const_ranges);
    } else if (T == VULTEN_INT64) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_int64_t,
                          nullptr, push_const_ranges);
    } else if (T == VULTEN_UINT64) {
      vulten_pipeline =
          create_pipeline(pipe_string, 2, shader::AddSubInPlace_uint64_t,
                          nullptr, push_const_ranges);
    } else {
      throw std::runtime_error("Error unsuported type in Assign_add_sub: " +
                               std::to_string(T));
    }
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Assign_add_sub_op pipeline " +
                     pipe_string)
    vulten_pipeline = pipelines[pipe_string];
  }

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);
  vk::DescriptorBufferInfo value_buffer_info(value.buffer->vk_buffer, 0,
                                             value.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {vulten_pipeline->descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &input_buffer_info},
      {vulten_pipeline->descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &value_buffer_info},
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
  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t),
                         &op);
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

template <Data_type T>
Assign_add_sub_op<T>::~Assign_add_sub_op() {
  VULTEN_LOG_DEBUG("Freeing vulten_ops::Assign_add_sub_op<" +
                   Data_type_to_str(T) + ">")
}

}  // namespace vulten_ops