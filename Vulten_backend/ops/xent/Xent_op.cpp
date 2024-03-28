#include "Xent_op.h"

#include <cmath>

#include "Xent_shader.h"

namespace vulten_ops {
namespace xent {

void run_op(vulten_backend::Instance *inst, Data_type dt, Data_type dt_labels,
            Vulten_tensor scratch, Vulten_tensor backprop, Vulten_tensor labels,
            Vulten_tensor output, uint32_t op) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Xent_op<" + Data_type_to_str(dt) + ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  std::string pipe_string = "Xent_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Xent_op pipeline " + pipe_string)

    xent_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(xent_shader::Spec_cons, localX),
         sizeof(xent_shader::Spec_cons)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(),
                                     sizeof(xent_shader::Spec_cons), &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0,
         sizeof(xent_shader::Push_const)}};

    xent_shader::Generate_xent_shader_info generate_xent_shader_info{dt,
                                                                     dt_labels};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS,
                                        vk::DescriptorType::eStorageBuffer);
    vulten_pipeline = inst->create_pipeline(
        pipe_string, buffer_types,
        xent_shader::generate_xent_shader(generate_xent_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Xent_op pipeline " + pipe_string)
  }

  vulten_backend::Descriptor_set_alloc descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS,
                                vulten_pipeline->descriptor_set_layout);

  vk::DescriptorBufferInfo scratch_buffer_info(scratch.buffer->vk_buffer, 0,
                                               scratch.buffer->buffer_size);
  vk::DescriptorBufferInfo backprop_buffer_info(backprop.buffer->vk_buffer, 0,
                                                backprop.buffer->buffer_size);
  vk::DescriptorBufferInfo labels_buffer_info(labels.buffer->vk_buffer, 0,
                                              labels.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &scratch_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &backprop_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 2, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &labels_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 3, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  auto [cmd_buff_res, cmd_buffs] =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);
  RES_CHECK_SUCCESS_ONLY(cmd_buff_res)
  vk::CommandBuffer cmd_buff = cmd_buffs.front();

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  RES_CHECK_SUCCESS_ONLY(cmd_buff.begin(cmd_buff_begin_info));
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        vulten_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets
  uint32_t threads = std::ceil(
      float(backprop.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

  xent_shader::Push_const push_const{uint32_t(backprop.dims[1]), op};
  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0,
                         sizeof(xent_shader::Push_const), &push_const);
  cmd_buff.dispatch(threads, 1, 1);
  RES_CHECK_SUCCESS_ONLY(cmd_buff.end());

  auto [fence_res, fence] =
      inst->logical_dev.createFence(vk::FenceCreateInfo());
  RES_CHECK_SUCCESS_ONLY(fence_res)

  vk::SubmitInfo SubmitInfo({}, {}, cmd_buffs);
  RES_CHECK_SUCCESS_ONLY(
      queue_alloc.queue->vk_queue.submit({SubmitInfo}, fence));
  RES_CHECK_SUCCESS_ONLY(
      inst->logical_dev.waitForFences({fence},         // List of fences
                                      true,            // Wait All
                                      uint64_t(-1)));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buffs);
}

}  // namespace xent
}  // namespace vulten_ops