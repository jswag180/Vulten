#include "Bias_add_grad.h"

#include <cmath>

#include "shaders/headers/BiasAddGrad/BiasAddGrad.comp.h"

#define NUM_BUFFERS 2
#define NUM_SETS 1

namespace vulten_ops {

Bias_add_grad_op::Bias_add_grad_op(vulten_backend::Instance *inst)
    : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Bias_add_grad_op")
}

void Bias_add_grad_op::run_op(Data_type dt, Vulten_tensor input,
                              Channel_format channel_format,
                              Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Bias_add_grad_op<" +
                   Data_type_to_str(dt) + ">")
  inst->main_queue_mutex.lock();

  vulten_backend::Device_propertys dev_props =
      vulten_backend::Device_propertys();
  std::string pipe_string =
      "Bias_add_grad_" + Data_type_to_str(dt) + "_" +
      std::to_string((*dev_props.devices)[inst->device_num]
                         .props.limits.maxComputeWorkGroupInvocations) +
      "_" +
      std::to_string((*dev_props.devices)[inst->device_num].subgroupSize) +
      "_" + std::to_string(channel_format) + "_" +
      std::to_string(input.dims[0]) + "_" + std::to_string(input.dims[1]) +
      "_" + std::to_string(input.dims[2]) + "_" + std::to_string(input.dims[3]);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Bias_add_grad_op pipeline " +
                     pipe_string)

    struct Spec {
      uint32_t local_size_x;
      uint32_t subgroup_size;
      uint32_t channel_format;
      uint32_t n, h, w, c;
    } spec;

    spec.local_size_x = (*dev_props.devices)[inst->device_num]
                            .props.limits.maxComputeWorkGroupInvocations;
    spec.subgroup_size = (*dev_props.devices)[inst->device_num].subgroupSize;
    spec.channel_format = channel_format;

    spec.n = input.dims[0];
    spec.h = input.dims[1];
    spec.w = input.dims[2];
    spec.c = input.dims[3];

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(uint32_t)},
        {1, offsetof(Spec, subgroup_size), sizeof(uint32_t)},
        {2, offsetof(Spec, channel_format), sizeof(uint32_t)},
        {3, offsetof(Spec, n), sizeof(uint32_t)},
        {4, offsetof(Spec, h), sizeof(uint32_t)},
        {5, offsetof(Spec, w), sizeof(uint32_t)},
        {6, offsetof(Spec, c), sizeof(uint32_t)}};
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}};

    std::vector<Data_type> type_chain = {dt};
    vulten_pipeline =
        create_pipeline(pipe_string, 2, BiasAddGrad_comp, type_chain.data(),
                        type_chain.size(), &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Bias_add_grad_op pipeline " +
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
      descriptor_pool, 1, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &input_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary,
      input.dims[channel_format == Channel_format::NHWC ? 3 : 1]);
  std::vector<vk::CommandBuffer> cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  uint32_t threads = 0;
  float local_x = float((*dev_props.devices)[inst->device_num]
                            .props.limits.maxComputeWorkGroupInvocations);
  if (channel_format == Channel_format::NHWC) {
    threads = uint32_t(std::ceil(
        uint32_t(input.get_total_elements() / input.dims[3]) / local_x));
  } else {
    threads =
        uint32_t(std::ceil(uint32_t(input.dims[2] * input.dims[3]) / local_x));
  }

  for (uint32_t i = 0; i < cmd_buffs.size(); i++) {
    cmd_buffs[i].begin(cmd_buff_begin_info);
    cmd_buffs[i].bindPipeline(vk::PipelineBindPoint::eCompute,
                              vulten_pipeline->pipeline);
    cmd_buffs[i].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,   // Bind point
        vulten_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                 // First descriptor set
        {descriptor_set},                  // List of descriptor sets
        {});                               // Dynamic offsets

    cmd_buffs[i].pushConstants(vulten_pipeline->pipeline_layout,
                               vk::ShaderStageFlagBits::eCompute, 0, sizeof(i),
                               &i);

    cmd_buffs[i].dispatch(threads, 1, 1);
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
  inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

Bias_add_grad_op::~Bias_add_grad_op() {
  VULTEN_LOG_DEBUG("Freeing vulten_ops::Bias_add_grad_op")
}

}  // namespace vulten_ops