#include "Cast_op.h"

#include "../../shaders/headers/Cast/Cast.h"

namespace vulten_ops {

#define DEFINE_CAST(X)                      \
  template class Cast_op<X, VULTEN_FLOAT>;  \
  template class Cast_op<X, VULTEN_INT32>;  \
  template class Cast_op<X, VULTEN_UINT32>; \
  template class Cast_op<X, VULTEN_INT64>;  \
  template class Cast_op<X, VULTEN_UINT64>; \
  template class Cast_op<X, VULTEN_INT8>;   \
  template class Cast_op<X, VULTEN_UINT8>;  \
  template class Cast_op<X, VULTEN_DOUBLE>; \
  template class Cast_op<X, VULTEN_FLOAT16>;

VULTEN_DEFINE_BASIC_TYPES(DEFINE_CAST)

#define CREATE_PIPELINES(SRC_TYPE, SRC_TYPE_NAME)                              \
  if (SRC == SRC_TYPE && DST == VULTEN_FLOAT) {                                \
    vulten_pipeline =                                                          \
        create_pipeline(pipe_string, 2, shader::Cast_##SRC_TYPE_NAME##_float); \
  } else if (SRC == SRC_TYPE && DST == VULTEN_INT32) {                         \
    vulten_pipeline =                                                          \
        create_pipeline(pipe_string, 2, shader::Cast_##SRC_TYPE_NAME##_int);   \
  } else if (SRC == SRC_TYPE && DST == VULTEN_UINT32) {                        \
    vulten_pipeline =                                                          \
        create_pipeline(pipe_string, 2, shader::Cast_##SRC_TYPE_NAME##_uint);  \
  } else if (SRC == SRC_TYPE && DST == VULTEN_INT64) {                         \
    vulten_pipeline = create_pipeline(pipe_string, 2,                          \
                                      shader::Cast_##SRC_TYPE_NAME##_int64_t); \
  } else if (SRC == SRC_TYPE && DST == VULTEN_UINT64) {                        \
    vulten_pipeline = create_pipeline(                                         \
        pipe_string, 2, shader::Cast_##SRC_TYPE_NAME##_uint64_t);              \
  } else if (SRC == SRC_TYPE && DST == VULTEN_INT8) {                          \
    vulten_pipeline = create_pipeline(pipe_string, 2,                          \
                                      shader::Cast_##SRC_TYPE_NAME##_int8_t);  \
  } else if (SRC == SRC_TYPE && DST == VULTEN_UINT8) {                         \
    vulten_pipeline = create_pipeline(pipe_string, 2,                          \
                                      shader::Cast_##SRC_TYPE_NAME##_uint8_t); \
  } else if (SRC == SRC_TYPE && DST == VULTEN_DOUBLE) {                        \
    vulten_pipeline = create_pipeline(pipe_string, 2,                          \
                                      shader::Cast_##SRC_TYPE_NAME##_double);  \
  } else if (SRC == SRC_TYPE && DST == VULTEN_FLOAT16) {                       \
    vulten_pipeline = create_pipeline(                                         \
        pipe_string, 2, shader::Cast_##SRC_TYPE_NAME##_float16_t);             \
  }

template <Data_type SRC, Data_type DST>
Cast_op<SRC, DST>::Cast_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Cast_op<" + Data_type_to_str(SRC) +
                   ", " + Data_type_to_str(DST) + ">")
}

template <Data_type SRC, Data_type DST>
void Cast_op<SRC, DST>::run_op(Vulten_tensor input, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Cast_op<" + Data_type_to_str(SRC) +
                   ", " + Data_type_to_str(DST) + ">")
  inst->main_queue_mutex.lock();

  std::string pipe_string =
      "Cast_" + Data_type_to_str(SRC) + "_" + Data_type_to_str(DST);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Cast_op pipeline " + pipe_string)

    CREATE_PIPELINES(VULTEN_FLOAT, float)
    CREATE_PIPELINES(VULTEN_INT32, int)
    CREATE_PIPELINES(VULTEN_UINT32, uint)
    CREATE_PIPELINES(VULTEN_INT64, int64_t)
    CREATE_PIPELINES(VULTEN_UINT64, uint64_t)
    CREATE_PIPELINES(VULTEN_INT8, int8_t)
    CREATE_PIPELINES(VULTEN_UINT8, uint8_t)
    CREATE_PIPELINES(VULTEN_DOUBLE, double)
    CREATE_PIPELINES(VULTEN_FLOAT16, float16_t)
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Cast_op pipeline " + pipe_string)
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

template <Data_type SRC, Data_type DST>
Cast_op<SRC, DST>::~Cast_op() {
  VULTEN_LOG_DEBUG("Freeing vulten_ops::Cast_op<" + Data_type_to_str(SRC) +
                   ", " + Data_type_to_str(DST) + ">")
}

}  // namespace vulten_ops