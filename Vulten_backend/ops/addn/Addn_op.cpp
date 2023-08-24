#include "Addn_op.h"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "../assign_add_sub/Assign_add_sub_op.h"
#include "Vulten_backend/Vulten_backend.h"

namespace vulten_ops {
namespace addn {

void run_op(vulten_backend::Instance *inst, Data_type dt,
            std::vector<Vulten_tensor> &inputs, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Addn_op<" + Data_type_to_str(dt) + ">")
  inst->main_queue_mutex.lock();

  vulten_backend::Vulten_pipeline *vulten_pipeline =
      assign_add_sub::get_assign_add_sub_pipeline(inst, dt);

  uint32_t num_sets = inputs.size() - 1;

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer,
      assign_add_sub::NUM_BUFFERS * num_sets);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, num_sets,
      descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  std::vector<vk::DescriptorSetLayout> descriptor_set_layouts(
      num_sets, vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, num_sets, descriptor_set_layouts.data());
  std::vector<vk::DescriptorSet> descriptor_sets =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info);

  std::vector<vk::DescriptorBufferInfo> buffer_info =
      std::vector<vk::DescriptorBufferInfo>(
          assign_add_sub::NUM_BUFFERS * num_sets, vk::DescriptorBufferInfo());
  for (uint32_t i = 0; i < buffer_info.size();
       i += assign_add_sub::NUM_BUFFERS) {
    buffer_info[i].setBuffer(output.buffer->vk_buffer);
    buffer_info[i].setOffset(0);
    buffer_info[i].setRange(output.buffer->buffer_size);

    buffer_info[i + 1].setBuffer(
        inputs[(i / assign_add_sub::NUM_BUFFERS) + 1].buffer->vk_buffer);
    buffer_info[i + 1].setOffset(0);
    buffer_info[i + 1].setRange(
        inputs[(i / assign_add_sub::NUM_BUFFERS) + 1].buffer->buffer_size);
  }

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
      std::vector<vk::WriteDescriptorSet>(num_sets *
                                          assign_add_sub::NUM_BUFFERS);
  for (uint32_t i = 0; i < num_sets; i++) {
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDstSet(
        descriptor_sets[i]);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDstBinding(0);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDstArrayElement(0);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDescriptorCount(1);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDescriptorType(
        vk::DescriptorType::eStorageBuffer);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setBufferInfo(
        buffer_info[i * assign_add_sub::NUM_BUFFERS]);

    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1].setDstSet(
        descriptor_sets[i]);
    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1].setDstBinding(1);
    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1]
        .setDstArrayElement(0);
    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1]
        .setDescriptorCount(1);
    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1]
        .setDescriptorType(vk::DescriptorType::eStorageBuffer);
    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1].setBufferInfo(
        buffer_info[(i * assign_add_sub::NUM_BUFFERS) + 1]);
  }
  inst->logical_dev.updateDescriptorSets(writeDescriptorSets, {});

  int32_t op = ADD;

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
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        vulten_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_sets[0]},              // List of descriptor sets
      {});                               // Dynamic offsets

  vk::BufferCopy buff_copy(0, 0, inputs[0].buffer->buffer_size);
  cmd_buff.copyBuffer(inputs[0].buffer->vk_buffer, output.buffer->vk_buffer,
                      buff_copy);

  cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlags(), mem_bar, nullptr, nullptr);

  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t),
                         &op);
  cmd_buff.dispatch(uint32_t(inputs[0].get_total_elements()), 1, 1);

  for (uint32_t i = 2; i < inputs.size(); i++) {
    cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             vk::DependencyFlags(), mem_bar, nullptr, nullptr);

    cmd_buff.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,   // Bind point
        vulten_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                 // First descriptor set
        {descriptor_sets[i - 1]},          // List of descriptor sets
        {});                               // Dynamic offsets

    cmd_buff.dispatch(uint32_t(inputs[0].get_total_elements()), 1, 1);
  }

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
  inst->logical_dev.freeDescriptorSets(descriptor_pool, num_sets,
                                       descriptor_sets.data());
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

}  // namespace addn
}  // namespace vulten_ops