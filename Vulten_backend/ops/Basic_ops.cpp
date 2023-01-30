#include "Basic_ops.h"

#include "../../shaders/headers/BasicOps/BasicOps.comp.h"

namespace vulten_ops {

struct Push_const {
  uint32_t batch_num;
};

std::string op_as_str(uint32_t op) {
  if (op == OP_MUL) {
    return "Mul";
  } else if (op == OP_ADD) {
    return "AddV2";
  } else if (op == OP_SUB) {
    return "Sub";
  } else if (op == OP_DIV) {
    return "Div";
  } else if (op == OP_DIV_NO_NAN) {
    return "DivNoNan";
  } else {
    return "INVALID";
  }
}

Basic_op::Basic_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Basic_op")
}

void Basic_op::run_op(Data_type dt, uint32_t op, Vulten_tensor x,
                      Vulten_tensor y, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Basic_op<" + Data_type_to_str(dt) +
                   ", " + op_as_str(op) + ">")
  inst->main_queue_mutex.lock();

  // A future optimization could be the check if x_dims and y_dims are swaped so
  // we could do y + x instead of x + y and avoid an extra pipeline
  std::string basic_pipe_string =
      "Basic_" + op_as_str(op) + "_" + Data_type_to_str(dt) + "_" +
      std::to_string(output.dims[0]) + "_" + std::to_string(x.dims[0]) + "_" +
      std::to_string(x.dims[1]) + "_" + std::to_string(x.dims[2]) + "_" +
      std::to_string(x.dims[3]) + "_" + std::to_string(y.dims[0]) + "_" +
      std::to_string(y.dims[1]) + "_" + std::to_string(y.dims[2]) + "_" +
      std::to_string(y.dims[3]) + "_" + std::to_string(op);
  Vulten_pipeline *vulten_pipeline = nullptr;
  if (!is_pipeline_cached(basic_pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Basic_op<" + Data_type_to_str(dt) +
                     ", " + op_as_str(op) + "> " + basic_pipe_string)

    struct Spec {
      uint32_t max_batches;

      uint32_t x_size_1;
      uint32_t x_size_2;
      uint32_t x_size_3;
      uint32_t x_size_4;

      uint32_t y_size_1;
      uint32_t y_size_2;
      uint32_t y_size_3;
      uint32_t y_size_4;

      uint32_t op;
    } spec;

    spec.max_batches = output.dims[0];

    spec.x_size_1 = x.dims[0];
    spec.x_size_2 = x.dims[1];
    spec.x_size_3 = x.dims[2];
    spec.x_size_4 = x.dims[3];

    spec.y_size_1 = y.dims[0];
    spec.y_size_2 = y.dims[1];
    spec.y_size_3 = y.dims[2];
    spec.y_size_4 = y.dims[3];

    spec.op = op;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(uint32_t)},
        {1, offsetof(Spec, x_size_1), sizeof(uint32_t)},
        {2, offsetof(Spec, x_size_2), sizeof(uint32_t)},
        {3, offsetof(Spec, x_size_3), sizeof(uint32_t)},
        {4, offsetof(Spec, x_size_4), sizeof(uint32_t)},
        {5, offsetof(Spec, y_size_1), sizeof(uint32_t)},
        {6, offsetof(Spec, y_size_2), sizeof(uint32_t)},
        {7, offsetof(Spec, y_size_3), sizeof(uint32_t)},
        {8, offsetof(Spec, y_size_4), sizeof(uint32_t)},
        {9, offsetof(Spec, op), sizeof(uint32_t)}};
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    std::vector<Data_type> type_chain = {dt};
    vulten_pipeline =
        create_pipeline(basic_pipe_string, 3, BasicOps_comp, type_chain.data(),
                        type_chain.size(), &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Basic_op<" +
                     Data_type_to_str(dt) + ", " + op_as_str(op) + "> " +
                     basic_pipe_string)
    vulten_pipeline = pipelines[basic_pipe_string];
  }

  vk::DescriptorBufferInfo x_buffer_info(x.buffer->vk_buffer, 0,
                                         x.buffer->buffer_size);
  vk::DescriptorBufferInfo y_buffer_info(y.buffer->vk_buffer, 0,
                                         y.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);
  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {vulten_pipeline->descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &x_buffer_info},
      {vulten_pipeline->descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &y_buffer_info},
      {vulten_pipeline->descriptor_set, 2, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary, output.dims[0]);
  std::vector<vk::CommandBuffer> cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  for (uint32_t i = 0; i < output.dims[0]; i++) {
    cmd_buffs[i].begin(cmd_buff_begin_info);
    cmd_buffs[i].bindPipeline(vk::PipelineBindPoint::eCompute,
                              vulten_pipeline->pipeline);
    cmd_buffs[i].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,    // Bind point
        vulten_pipeline->pipeline_layout,   // Pipeline Layout
        0,                                  // First descriptor set
        {vulten_pipeline->descriptor_set},  // List of descriptor sets
        {});                                // Dynamic offsets

    Push_const push_data = Push_const();
    push_data.batch_num = i;
    cmd_buffs[i].pushConstants(vulten_pipeline->pipeline_layout,
                               vk::ShaderStageFlagBits::eCompute, 0,
                               sizeof(push_data), &push_data);

    cmd_buffs[i].dispatch(uint32_t(output.dims[1]), uint32_t(output.dims[2]),
                          uint32_t(output.dims[3]));
    cmd_buffs[i].end();
  }

  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());

  vk::SubmitInfo SubmitInfo(0,                  // Num Wait Semaphores
                            nullptr,            // Wait Semaphores
                            nullptr,            // Pipeline Stage Flags
                            cmd_buffs.size(),   // Num Command Buffers
                            cmd_buffs.data());  // List of command buffers
  inst->main_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(inst->cmd_pool, cmd_buffs);
  inst->main_queue_mutex.unlock();
}

Basic_op::~Basic_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::Basic_op") }

}  // namespace vulten_ops
