#include "Resource_apply_adam_op.h"

#include "shaders/headers/ApplyAdam/ApplyAdam.comp.h"

namespace vulten_ops {

Resource_apply_adam_op::Resource_apply_adam_op(vulten_backend::Instance *inst)
    : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Resource_apply_adam_op")
}

void Resource_apply_adam_op::run_op(Data_type dt, Vulten_tensor var,
                                    Vulten_tensor m, Vulten_tensor v,
                                    Vulten_tensor beta1_power,
                                    Vulten_tensor beta2_power, Vulten_tensor lr,
                                    Vulten_tensor beta1, Vulten_tensor beta2,
                                    Vulten_tensor epsilon, Vulten_tensor grad,
                                    bool use_nesterov) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Resource_apply_adam_op<" +
                   Data_type_to_str(dt) + ">")
  inst->main_queue_mutex.lock();

  std::string pipe_string = "Resource_apply_adam_" + Data_type_to_str(dt) +
                            "_" + std::to_string(use_nesterov);
  Vulten_pipeline *vulten_pipeline = nullptr;

  if (!is_pipeline_cached(pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Resource_apply_adam_op pipeline " +
                     pipe_string)

    struct Spec_data {
      bool nesterov;
    } spec_data;

    spec_data.nesterov = use_nesterov;

    std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(spec_data.nesterov)}};
    vk::SpecializationInfo spec_info(1, specs.data(), sizeof(spec_data),
                                     &spec_data);

    std::vector<Data_type> type_chain = {dt};
    vulten_pipeline =
        create_pipeline(pipe_string, 10, ApplyAdam_comp, type_chain.data(),
                        type_chain.size(), &spec_info);
  } else {
    VULTEN_LOG_DEBUG(
        "Using cached vulten_ops::Resource_apply_adam_op pipeline " +
        pipe_string)
    vulten_pipeline = pipelines[pipe_string];
  }

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer, 10);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, 1, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo var_buffer_info(var.buffer->vk_buffer, 0,
                                           var.buffer->buffer_size);
  vk::DescriptorBufferInfo m_buffer_info(m.buffer->vk_buffer, 0,
                                         m.buffer->buffer_size);
  vk::DescriptorBufferInfo v_buffer_info(v.buffer->vk_buffer, 0,
                                         v.buffer->buffer_size);
  vk::DescriptorBufferInfo beta1_power_buffer_info(
      beta1_power.buffer->vk_buffer, 0, beta1_power.buffer->buffer_size);
  vk::DescriptorBufferInfo beta2_power_buffer_info(
      beta2_power.buffer->vk_buffer, 0, beta2_power.buffer->buffer_size);
  vk::DescriptorBufferInfo lr_buffer_info(lr.buffer->vk_buffer, 0,
                                          lr.buffer->buffer_size);
  vk::DescriptorBufferInfo beta1_buffer_info(beta1.buffer->vk_buffer, 0,
                                             beta1.buffer->buffer_size);
  vk::DescriptorBufferInfo beta2_buffer_info(beta2.buffer->vk_buffer, 0,
                                             beta2.buffer->buffer_size);
  vk::DescriptorBufferInfo epsilon_buffer_info(epsilon.buffer->vk_buffer, 0,
                                               epsilon.buffer->buffer_size);
  vk::DescriptorBufferInfo grad_buffer_info(grad.buffer->vk_buffer, 0,
                                            grad.buffer->buffer_size);

  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &var_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &m_buffer_info},
      {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &v_buffer_info},
      {descriptor_set, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &beta1_power_buffer_info},
      {descriptor_set, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &beta2_power_buffer_info},
      {descriptor_set, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &lr_buffer_info},
      {descriptor_set, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &beta1_buffer_info},
      {descriptor_set, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &beta2_buffer_info},
      {descriptor_set, 8, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &epsilon_buffer_info},
      {descriptor_set, 9, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &grad_buffer_info},
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
  cmd_buff.dispatch(uint32_t(var.get_total_elements()), 1, 1);
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

Resource_apply_adam_op::~Resource_apply_adam_op() {
  VULTEN_LOG_DEBUG("Freeing vulten_ops::Resource_apply_adam_op")
}

}  // namespace vulten_ops