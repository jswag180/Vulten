#include "Addn_op.h"

#include <vulkan/vulkan_core.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "../assign_add_sub/Assign_add_sub_op.h"
#include "Vulten_backend/Vulten_backend.h"

namespace vulten_ops {
namespace addn {

void run_op(vulten_backend::Instance *inst, Data_type dt,
            std::vector<Vulten_tensor> &inputs, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Addn_op<" + Data_type_to_str(dt) + ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  vulten_backend::Vulten_pipeline *vulten_pipeline =
      assign_add_sub::get_assign_add_sub_pipeline(inst, dt);

  uint32_t num_sets = inputs.size() - 1;

  auto descriptor_sets =
      std::vector<std::unique_ptr<vulten_backend::Descriptor_set_alloc>>(
          num_sets);
  for (uint32_t i = 0; i < descriptor_sets.size(); i++) {
    descriptor_sets[i] = std::make_unique<vulten_backend::Descriptor_set_alloc>(
        inst->get_descriptor_sets(assign_add_sub::NUM_BUFFERS,
                                  vulten_pipeline->descriptor_set_layout));
  }

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
        descriptor_sets[i]->descriptor_set->vk_descriptor_set);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDstBinding(0);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDstArrayElement(0);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDescriptorCount(1);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setDescriptorType(
        vk::DescriptorType::eStorageBuffer);
    writeDescriptorSets[i * assign_add_sub::NUM_BUFFERS].setBufferInfo(
        buffer_info[i * assign_add_sub::NUM_BUFFERS]);

    writeDescriptorSets[(i * assign_add_sub::NUM_BUFFERS) + 1].setDstSet(
        descriptor_sets[i]->descriptor_set->vk_descriptor_set);
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
  cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute,
                        vulten_pipeline->pipeline);
  cmd_buff.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_sets[0]
           ->descriptor_set->vk_descriptor_set},  // List of descriptor sets
      {});                                        // Dynamic offsets

  vk::BufferCopy buff_copy(0, 0, inputs[0].buffer->buffer_size);
  cmd_buff.copyBuffer(inputs[0].buffer->vk_buffer, output.buffer->vk_buffer,
                      buff_copy);

  cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlags(), mem_bar, nullptr, nullptr);

  cmd_buff.pushConstants(vulten_pipeline->pipeline_layout,
                         vk::ShaderStageFlagBits::eCompute, 0, sizeof(int32_t),
                         &op);
  uint32_t threads = std::ceil(
      float(inputs[0].get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
  cmd_buff.dispatch(threads, 1, 1);

  for (uint32_t i = 2; i < inputs.size(); i++) {
    cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             vk::DependencyFlags(), mem_bar, nullptr, nullptr);

    cmd_buff.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,   // Bind point
        vulten_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                 // First descriptor set
        {descriptor_sets[i - 1]
             ->descriptor_set->vk_descriptor_set},  // List of descriptor sets
        {});                                        // Dynamic offsets

    uint32_t threads = std::ceil(
        float(inputs[0].get_total_elements()) /
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
    cmd_buff.dispatch(threads, 1, 1);
  }

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

}  // namespace addn
}  // namespace vulten_ops