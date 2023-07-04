#include "Softmax_op.h"

#include <cmath>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "Vulten_backend/ops/Vulten_backend_ops.h"
#include "shaders/headers/BatchAdd/BatchAdd.comp.h"
#include "shaders/headers/Exp/Exp.comp.h"
#include "shaders/headers/Softmax/Softmax.comp.h"

#define NUM_BUFFERS_BATCHADD 2
#define NUM_BUFFERS_EXP 2
#define NUM_BUFFERS_SOFTMAX 3
#define NUM_SETS 3

namespace vulten_ops {

Softmax_op::Softmax_op(vulten_backend::Instance* inst)
    : Vulten_op(inst){VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op")}

      vulten_ops::Vulten_pipeline
      *
      Softmax_op::get_exp_pipeline(Data_type dt) {
  std::string pipe_string = "Exp_" + Data_type_to_str(dt);

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op pipeline " + pipe_string)

    struct Spec {
      uint32_t localX;
    } spec;

    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(Spec, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    std::vector<Data_type> type_chain = {dt};
    return create_pipeline(pipe_string, NUM_BUFFERS_EXP, Exp_comp,
                           type_chain.data(), type_chain.size(), &spec_info);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Softmax_op pipeline " +
                     pipe_string)
    return pipelines[pipe_string];
  }
}

vulten_ops::Vulten_pipeline* Softmax_op::get_batchAdd_pipeline(Data_type dt) {
  std::string pipe_string = "BatchAdd_" + Data_type_to_str(dt);

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op pipeline " + pipe_string)

    struct Spec {
      uint32_t localX;
    } spec;

    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(Spec, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    std::vector<Data_type> type_chain = {dt};
    return create_pipeline(pipe_string, NUM_BUFFERS_BATCHADD, BatchAdd_comp,
                           type_chain.data(), type_chain.size(), &spec_info,
                           push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Softmax_op pipeline " +
                     pipe_string)
    return pipelines[pipe_string];
  }
}

vulten_ops::Vulten_pipeline* Softmax_op::get_softmax_pipeline(Data_type dt) {
  std::string pipe_string = "Softmax_" + Data_type_to_str(dt);

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Softmax_op pipeline " + pipe_string)

    struct Spec {
      uint32_t localX;
    } spec;

    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(Spec, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    std::vector<Data_type> type_chain = {dt};
    return create_pipeline(pipe_string, NUM_BUFFERS_SOFTMAX, Softmax_comp,
                           type_chain.data(), type_chain.size(), &spec_info,
                           push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Softmax_op pipeline " +
                     pipe_string)
    return pipelines[pipe_string];
  }
}

void Softmax_op::run_op(Data_type dt, Vulten_tensor input,
                        Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Softmax_op<" + Data_type_to_str(dt) +
                   ">")
  inst->main_queue_mutex.lock();

  vulten_ops::Vulten_pipeline* exp_pipeline = get_exp_pipeline(dt);
  vulten_ops::Vulten_pipeline* batchAdd_pipeline = get_batchAdd_pipeline(dt);
  vulten_ops::Vulten_pipeline* softmax_pipeline = get_softmax_pipeline(dt);

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer,
      NUM_BUFFERS_BATCHADD * NUM_BUFFERS_EXP * NUM_BUFFERS_SOFTMAX);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, NUM_SETS,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  std::array<vk::DescriptorSetLayout, NUM_SETS> descriptor_set_layouts = {
      exp_pipeline->descriptor_set_layout,
      batchAdd_pipeline->descriptor_set_layout,
      softmax_pipeline->descriptor_set_layout};
  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, NUM_SETS, descriptor_set_layouts.data());
  std::vector<vk::DescriptorSet> descriptor_sets =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info);

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
      {descriptor_sets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &input_buffer_info},
      {descriptor_sets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &exp_buffer_info},

      {descriptor_sets[1], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &exp_buffer_info},
      {descriptor_sets[1], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &exp_sum_buffer_info},

      {descriptor_sets[2], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &exp_buffer_info},
      {descriptor_sets[2], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &exp_sum_buffer_info},
      {descriptor_sets[2], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  vk::MemoryBarrier mem_bar = vk::MemoryBarrier(
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);

  cmd_buff.begin(cmd_buff_begin_info);

  // Exp
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        exp_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(vk::PipelineBindPoint::eCompute,  // Bind point
                              exp_pipeline->pipeline_layout,  // Pipeline Layout
                              0,                     // First descriptor set
                              {descriptor_sets[0]},  // List of descriptor sets
                              {});                   // Dynamic offsets
  uint32_t threads = std::ceil(
      float(input.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

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
      {descriptor_sets[1]},                // List of descriptor sets
      {});                                 // Dynamic offsets

  uint32_t num_logits = uint32_t(input.dims[1]);
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
      {descriptor_sets[2]},               // List of descriptor sets
      {});                                // Dynamic offsets

  cmd_buff.pushConstants(softmax_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t),
                         &num_logits);

  threads = std::ceil(
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
  inst->main_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(inst->cmd_pool, cmd_buff);
  inst->logical_dev.freeDescriptorSets(descriptor_pool, NUM_SETS,
                                       descriptor_sets.data());
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

Softmax_op::~Softmax_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::Softmax_op") }

}  // namespace vulten_ops