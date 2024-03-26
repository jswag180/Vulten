#include "Softmax_op.h"

#include <cmath>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "../multiFunc/MultiFunc_op.h"
#include "../multiFunc/MultiFunc_shader.h"
#include "BatchAdd_shader.h"
#include "Softmax_shader.h"
#include "Vulten_backend/ops/Vulten_backend_ops.h"

#define NUM_BUFFERS_BATCHADD 2
#define NUM_BUFFERS_SOFTMAX 3
#define NUM_SETS 3

namespace vulten_ops {
namespace softmax {

vulten_backend::Vulten_pipeline* get_batchAdd_pipeline(
    vulten_backend::Instance* inst, Data_type dt) {
  std::string pipe_string = "BatchAdd_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline* vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op pipeline " + pipe_string)

    batchAdd_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(batchAdd_shader::Spec_cons, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    batchAdd_shader::Generate_batchAdd_shader_info
        generate_batchAdd_shader_info{dt};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS_BATCHADD,
                                        vk::DescriptorType::eStorageBuffer);
    return inst->create_pipeline(pipe_string, buffer_types,
                                 batchAdd_shader::generate_batchAdd_shader(
                                     generate_batchAdd_shader_info),
                                 &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Softmax_op pipeline " +
                     pipe_string)
    return vulten_pipeline;
  }
}

vulten_backend::Vulten_pipeline* get_softmax_pipeline(
    vulten_backend::Instance* inst, Data_type dt) {
  std::string pipe_string = "Softmax_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline* vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op pipeline " + pipe_string)

    softmax_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(softmax_shader::Spec_cons, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(),
                                     sizeof(softmax_shader::Spec_cons), &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    softmax_shader::Generate_softmax_shader_info generate_softmax_shader_info{
        dt};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS_SOFTMAX,
                                        vk::DescriptorType::eStorageBuffer);
    return inst->create_pipeline(
        pipe_string, buffer_types,
        softmax_shader::generate_softmax_shader(generate_softmax_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Softmax_op pipeline " +
                     pipe_string)
    return vulten_pipeline;
  }
}

void run_op(vulten_backend::Instance* inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Softmax_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  vulten_backend::Vulten_pipeline* exp_pipeline =
      multiFunc::get_multiFunc_pipeline(inst, dt);
  vulten_backend::Vulten_pipeline* batchAdd_pipeline =
      get_batchAdd_pipeline(inst, dt);
  vulten_backend::Vulten_pipeline* softmax_pipeline =
      get_softmax_pipeline(inst, dt);

  vulten_backend::Descriptor_set_alloc exp_descriptor_set_alloc =
      inst->get_descriptor_sets(multiFunc::NUM_BUFFERS,
                                exp_pipeline->descriptor_set_layout);
  vulten_backend::Descriptor_set_alloc batchAdd_descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS_BATCHADD,
                                batchAdd_pipeline->descriptor_set_layout);
  vulten_backend::Descriptor_set_alloc softmax_descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS_SOFTMAX,
                                softmax_pipeline->descriptor_set_layout);

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);

  std::unique_ptr<vulten_backend::Device_buffer> exp_buffer =
      std::unique_ptr<vulten_backend::Device_buffer>(
          inst->create_device_buffer(input.buffer->buffer_size, false, false));
  vk::DescriptorBufferInfo exp_buffer_info(exp_buffer->vk_buffer, 0,
                                           exp_buffer->buffer_size);

  std::unique_ptr<vulten_backend::Device_buffer> exp_sum_buffer =
      std::unique_ptr<vulten_backend::Device_buffer>(
          inst->create_device_buffer(input.buffer->buffer_size, false, false));
  vk::DescriptorBufferInfo exp_sum_buffer_info(exp_sum_buffer->vk_buffer, 0,
                                               exp_sum_buffer->buffer_size);

  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {exp_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &input_buffer_info},
      {exp_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &exp_buffer_info},

      {batchAdd_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &exp_buffer_info},
      {batchAdd_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &exp_sum_buffer_info},

      {softmax_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &exp_buffer_info},
      {softmax_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &exp_sum_buffer_info},
      {softmax_descriptor_set_alloc.descriptor_set->vk_descriptor_set, 2, 0, 1,
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

  vk::MemoryBarrier mem_bar = vk::MemoryBarrier(
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);

  RES_CHECK_SUCCESS_ONLY(cmd_buff.begin(cmd_buff_begin_info));

  // Exp
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        exp_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,  // Bind point
      exp_pipeline->pipeline_layout,    // Pipeline Layout
      0,                                // First descriptor set
      {exp_descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets
  uint32_t threads = std::ceil(
      float(input.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

  multiFunc_shader::Push_const push_const{OP_EXP};
  cmd_buff.pushConstants(exp_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0,
                         sizeof(multiFunc_shader::Push_const), &push_const);
  cmd_buff.dispatch(threads, 1, 1);

  cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlags(), mem_bar, nullptr, nullptr);

  // Exp sum
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        batchAdd_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,     // Bind point
      batchAdd_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                   // First descriptor set
      {batchAdd_descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets

  uint32_t num_logits = uint32_t(input.dims[input.num_dims - 1]);
  cmd_buff.pushConstants(batchAdd_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t),
                         &num_logits);

  threads = std::ceil(
      (float(input.get_total_elements()) / num_logits) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

  cmd_buff.dispatch(threads, 1, 1);

  cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlags(), mem_bar, nullptr, nullptr);

  // Softmax
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        softmax_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,    // Bind point
      softmax_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                  // First descriptor set
      {softmax_descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets

  cmd_buff.pushConstants(softmax_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t),
                         &num_logits);

  threads = std::ceil(
      float(input.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

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

}  // namespace softmax
}  // namespace vulten_ops