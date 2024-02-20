#include "Broadcast_op.h"

#include <cmath>

#include "Broadcast_shader.h"
#include "Vulten_backend/Vulten_backend.h"
#include "Vulten_backend/Vulten_utills.h"

namespace vulten_ops {
namespace broadcast {

vulten_backend::Vulten_pipeline *get_broadcast_pipeline(
    vulten_backend::Instance *inst, Data_type dt) {
  std::string pipe_string = "Broadcast_" + Data_type_to_str(dt);
  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Broadcast_op<" +
                     Data_type_to_str(dt) + "> " + pipe_string)

    broadcast_shader::Spec_cons spec = broadcast_shader::Spec_cons();
    spec.local_x =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0,
         sizeof(broadcast_shader::Push_const)},
    };

    broadcast_shader::Generate_broadcast_shader_info
        generate_broadcast_shader_info{dt};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS,
                                        vk::DescriptorType::eStorageBuffer);
    return inst->create_pipeline(pipe_string, buffer_types,
                                 broadcast_shader::generate_broadcast_shader(
                                     generate_broadcast_shader_info),
                                 &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Broadcast_op<" +
                     Data_type_to_str(dt) + "> " + pipe_string)
    return vulten_pipeline;
  }
}

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Broadcast_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  vulten_backend::Vulten_pipeline *vulten_pipeline =
      get_broadcast_pipeline(inst, dt);

  vulten_backend::Descriptor_set_alloc descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS,
                                vulten_pipeline->descriptor_set_layout);

  vk::DescriptorBufferInfo input_buffer_info(input.buffer->vk_buffer, 0,
                                             input.buffer->buffer_size);

  std::vector<uint32_t> adj_strides =
      vulten_utills::calculate_adj_strides(input.dims, input.num_dims);
  std::vector<uint32_t> output_strides =
      vulten_utills::calculate_adj_strides(output.dims, output.num_dims);
  adj_strides.insert(adj_strides.end(), output_strides.begin(),
                     output_strides.end());
  auto strides_stageing = std::unique_ptr<vulten_backend::Host_mappable_buffer>(
      inst->create_host_mappable_buffer((uint8_t *)adj_strides.data(),
                                        sizeof(uint32_t) * adj_strides.size()));
  auto strides = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(strides_stageing->buffer_size, false, true));
  inst->copy_buffer(&queue_alloc, strides_stageing.get(), strides.get());
  vk::DescriptorBufferInfo strides_buffer_info(strides->vk_buffer, 0,
                                               strides->buffer_size);

  std::vector<uint32_t> dims_vec = std::vector<uint32_t>();
  for (uint32_t i = 0; i < input.num_dims; i++) {
    dims_vec.push_back(input.dims[i]);
  }
  auto dims_stageing = std::unique_ptr<vulten_backend::Host_mappable_buffer>(
      inst->create_host_mappable_buffer((uint8_t *)dims_vec.data(),
                                        sizeof(uint32_t) * dims_vec.size()));
  auto dims = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(dims_stageing->buffer_size, false, true));
  inst->copy_buffer(&queue_alloc, dims_stageing.get(), dims.get());
  vk::DescriptorBufferInfo dims_buffer_info(dims->vk_buffer, 0,
                                            dims->buffer_size);

  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);
  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 0, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &input_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 1, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &strides_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 2, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &dims_buffer_info},
      {descriptor_set_alloc.descriptor_set->vk_descriptor_set, 3, 0, 1,
       vk::DescriptorType::eStorageBuffer, nullptr, &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info).front();

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  cmd_buffs.begin(cmd_buff_begin_info);
  cmd_buffs.bindPipeline(vk::PipelineBindPoint::eCompute,
                         vulten_pipeline->pipeline);
  cmd_buffs.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_set_alloc.descriptor_set
           ->vk_descriptor_set},  // List of descriptor sets
      {});                        // Dynamic offsets

  broadcast_shader::Push_const pushes = {uint32_t(input.num_dims),
                                         uint32_t(output.num_dims)};
  cmd_buffs.pushConstants(vulten_pipeline->pipeline_layout,
                          vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(broadcast_shader::Push_const), &pushes);
  uint32_t threads = std::ceil(
      float(output.get_total_elements()) /
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);

  cmd_buffs.dispatch(threads, 1, 1);
  cmd_buffs.end();

  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());

  vk::SubmitInfo SubmitInfo(0,            // Num Wait Semaphores
                            nullptr,      // Wait Semaphores
                            nullptr,      // Pipeline Stage Flags
                            1,            // Num Command Buffers
                            &cmd_buffs);  // List of command buffers
  queue_alloc.queue->vk_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buffs);
}

}  // namespace broadcast
}  // namespace vulten_ops
