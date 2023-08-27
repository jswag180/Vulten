#include "Assign_add_sub_op.h"

#include <cmath>

#include "Assign_add_sub_shader.h"

namespace vulten_ops {
namespace assign_add_sub {

vulten_backend::Vulten_pipeline *get_assign_add_sub_pipeline(
    vulten_backend::Instance *inst, Data_type dt) {
  std::string pipe_string = "Assign_add_sub_" + Data_type_to_str(dt);

  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Assign_add_sub_op pipeline " +
                     pipe_string)

    assign_add_sub_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(assign_add_sub_shader::Spec_cons, localX),
         sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t)}};

    assign_add_sub_shader::Generate_assign_add_sub_shader_info
        generate_assign_add_sub_shader_info{dt};
    return inst->create_pipeline(
        pipe_string, NUM_BUFFERS,
        assign_add_sub_shader::generate_assign_add_sub_shader(
            generate_assign_add_sub_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Assign_add_sub_op pipeline " +
                     pipe_string)
    return vulten_pipeline;
  }
}

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor value, int op) {
  if (op == ADD) {
    VULTEN_LOG_DEBUG("Running vulten_ops::Assign_add_op<" +
                     Data_type_to_str(dt) + ">")
  } else if (op == SUB) {
    VULTEN_LOG_DEBUG("Running vulten_ops::Assign_sub_op<" +
                     Data_type_to_str(dt) + ">")
  }
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  vulten_backend::Vulten_pipeline *vulten_pipeline =
      get_assign_add_sub_pipeline(inst, dt);

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer, NUM_BUFFERS);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, NUM_SETS,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, 1, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);
  vk::DescriptorBufferInfo value_buffer_info(value.buffer->vk_buffer, 0,
                                             value.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &input_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &value_buffer_info},
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
  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t),
                         &op);
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
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
}

}  // namespace assign_add_sub
}  // namespace vulten_ops