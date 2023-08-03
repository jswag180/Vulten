#include "Pow_op.h"

#include <cmath>
#include <vulkan/vulkan_enums.hpp>

#include "Pow_shader.h"

#define NUM_BUFFERS 3
#define NUM_SETS 1

namespace vulten_ops {

Pow_op::Pow_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Pow_op")
}

void Pow_op::run_op(Data_type dt, uint32_t scalar, Vulten_tensor x,
                    Vulten_tensor y, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Pow_op<" + Data_type_to_str(dt) + ">")
  inst->main_queue_mutex.lock();

  std::string pipe_string = "Pow_" + Data_type_to_str(dt);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Pow_op pipeline " + pipe_string)

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

    Generate_pow_shader_info generate_pow_shader_info{dt};
    vulten_pipeline = create_pipeline(
        pipe_string, NUM_BUFFERS, generate_pow_shader(generate_pow_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Pow_op pipeline " + pipe_string)
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
      descriptor_pool, 1, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo x_buffer_info(x.buffer->vk_buffer, 0,
                                         x.buffer->buffer_size);
  vk::DescriptorBufferInfo y_buffer_info(y.buffer->vk_buffer, 0,
                                         y.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &x_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &y_buffer_info},
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

  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t),
                         &scalar);
  uint32_t threads = 0;
  if (scalar == 1) {
    threads = std::ceil(
        float(y.get_total_elements()) /
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
  } else {
    threads = std::ceil(
        float(x.get_total_elements()) /
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
  }

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
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

Pow_op::~Pow_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::Pow_op") }

}  // namespace vulten_ops