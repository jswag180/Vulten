#include "Reduce_op.h"

#include "Reduce_shader.h"
#include "Vulten_backend/Vulten_utills.h"

namespace vulten_ops {
namespace reduce {

std::string op_as_str(uint32_t op) {
  switch (op) {
    case OP_SUM:
      return "Sum";
      break;
    case OP_MAX:
      return "Max";
      break;
    case OP_MIN:
      return "Min";
      break;
    default:
      return "INVALID";
  }
}

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor input,
            std::vector<int32_t> &axis, Vulten_tensor output, uint32_t op) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Reduce_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  std::string pipe_string =
      "Reduce_" + Data_type_to_str(dt) + "_" +
      std::to_string(
          inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
  vulten_backend::Vulten_pipeline *vulten_pipeline =
      inst->get_cached_pipeline(pipe_string);
  if (vulten_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Reduce_op pipeline " + pipe_string)

    reduce_shader::Spec_cons spec;
    spec.localX =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(reduce_shader::Spec_cons, localX), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0,
         sizeof(reduce_shader::Push_const)}};

    reduce_shader::Generate_reduce_shader_info generate_reduce_shader_info{dt};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS,
                                        vk::DescriptorType::eStorageBuffer);
    vulten_pipeline = inst->create_pipeline(
        pipe_string, buffer_types,
        reduce_shader::generate_reduce_shader(generate_reduce_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Reduce_op pipeline " +
                     pipe_string)
  }

  uint32_t num_sets = axis.size();

  auto descriptor_sets =
      std::vector<std::unique_ptr<vulten_backend::Descriptor_set_alloc>>(
          num_sets);
  for (uint32_t i = 0; i < descriptor_sets.size(); i++) {
    descriptor_sets[i] = std::make_unique<vulten_backend::Descriptor_set_alloc>(
        inst->get_descriptor_sets(NUM_BUFFERS,
                                  vulten_pipeline->descriptor_set_layout));
  }

  std::vector<vk::Semaphore> axis_semaphores =
      std::vector<vk::Semaphore>(axis.size() - 1);
  for (uint32_t i = 0; i < axis_semaphores.size(); i++) {
    axis_semaphores[i] =
        inst->logical_dev.createSemaphore(vk::SemaphoreCreateInfo());
  }
  std::vector<vk::PipelineStageFlags> wait_stages =
      std::vector<vk::PipelineStageFlags>(
          axis.size() - 1, vk::PipelineStageFlagBits::eComputeShader);

  std::vector<std::unique_ptr<vulten_backend::Device_buffer>> scratch_buffers =
      std::vector<std::unique_ptr<vulten_backend::Device_buffer>>(axis.size() -
                                                                  1);
  uint32_t size_to_shave = 1;
  for (uint32_t i = 0; i < scratch_buffers.size(); i++) {
    size_to_shave *= uint32_t(input.dims[axis[i]]);
    scratch_buffers[i] = std::unique_ptr<vulten_backend::Device_buffer>(
        inst->create_device_buffer(input.buffer->buffer_size / size_to_shave,
                                   false, false));
  }

  std::vector<vk::DescriptorBufferInfo> buffer_info =
      std::vector<vk::DescriptorBufferInfo>(
          NUM_BUFFERS + scratch_buffers.size(), vk::DescriptorBufferInfo());

  buffer_info[0].setRange(input.buffer->buffer_size);
  buffer_info[0].setOffset(0);
  buffer_info[0].setBuffer(input.buffer->vk_buffer);
  for (uint32_t i = 1; i < buffer_info.size() - 1; i++) {
    buffer_info[i].setRange(scratch_buffers[i - 1].get()->buffer_size);
    buffer_info[i].setOffset(0);
    buffer_info[i].setBuffer(scratch_buffers[i - 1].get()->vk_buffer);
  }
  buffer_info[buffer_info.size() - 1].setRange(output.buffer->buffer_size);
  buffer_info[buffer_info.size() - 1].setOffset(0);
  buffer_info[buffer_info.size() - 1].setBuffer(output.buffer->vk_buffer);

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
      std::vector<vk::WriteDescriptorSet>(num_sets * NUM_BUFFERS);

  for (uint32_t i = 0; i < num_sets; i++) {
    writeDescriptorSets[i * NUM_BUFFERS].setDstSet(
        descriptor_sets[i]->descriptor_set->vk_descriptor_set);
    writeDescriptorSets[i * NUM_BUFFERS].setDstBinding(0);
    writeDescriptorSets[i * NUM_BUFFERS].setDstArrayElement(0);
    writeDescriptorSets[i * NUM_BUFFERS].setDescriptorCount(1);
    writeDescriptorSets[i * NUM_BUFFERS].setDescriptorType(
        vk::DescriptorType::eStorageBuffer);
    writeDescriptorSets[i * NUM_BUFFERS].setBufferInfo(buffer_info[i]);

    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setDstSet(
        descriptor_sets[i]->descriptor_set->vk_descriptor_set);
    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setDstBinding(1);
    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setDstArrayElement(0);
    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setDescriptorCount(1);
    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setDescriptorType(
        vk::DescriptorType::eStorageBuffer);
    writeDescriptorSets[(i * NUM_BUFFERS) + 1].setBufferInfo(
        buffer_info[i + 1]);
  }

  inst->logical_dev.updateDescriptorSets(writeDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary,
      axis.size());
  std::vector<vk::CommandBuffer> cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);
  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  std::vector<int64_t> dims = std::vector<int64_t>(input.num_dims);
  memcpy(dims.data(), input.dims, sizeof(int64_t) * input.num_dims);
  std::vector<uint32_t> adj_strides = std::vector<uint32_t>();

  uint32_t total_elements = 1;
  for (int64_t i : dims) {
    total_elements *= uint32_t(i);
  }

  for (uint32_t i = 0; i < cmd_buffs.size() - 1; i++) {
    cmd_buffs[i].begin(cmd_buff_begin_info);
    cmd_buffs[i].bindPipeline(vk::PipelineBindPoint::eCompute,
                              vulten_pipeline->pipeline);
    cmd_buffs[i].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,   // Bind point
        vulten_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                 // First descriptor set
        {descriptor_sets[i]
             ->descriptor_set->vk_descriptor_set},  // List of descriptor sets
        {});                                        // Dynamic offsets

    adj_strides = vulten_utills::calculate_adj_strides(dims);
    reduce_shader::Push_const push_const{
        uint32_t(dims[axis[i]]), uint32_t(adj_strides[axis[i]]),
        uint32_t(adj_strides[axis[i] + 1]), op};
    dims.erase(dims.begin() + axis[i]);
    cmd_buffs[i].pushConstants(vulten_pipeline->pipeline_layout,
                               vk::ShaderStageFlagBits::eCompute, 0,
                               sizeof(reduce_shader::Push_const), &push_const);

    total_elements /= push_const.axi_size;
    cmd_buffs[i].dispatch(
        (total_elements /
         inst->device_propertys.props.limits.maxComputeWorkGroupInvocations) +
            1,
        1, 1);

    cmd_buffs[i].end();

    vk::SubmitInfo SubmitInfo(
        i > 0 ? 1 : 0,                              // Num Wait Semaphores
        i > 0 ? &axis_semaphores[i - 1] : nullptr,  // Wait Semaphores
        wait_stages.data(),                         // Pipeline Stage Flags
        1,                                          // Num Command Buffers
        &cmd_buffs[i],                              // List of command buffers
        1, &axis_semaphores[i]);
    queue_alloc.queue->vk_queue.submit({SubmitInfo});
  }

  uint32_t final_cmd_buffer_ind = cmd_buffs.size() - 1;

  cmd_buffs[final_cmd_buffer_ind].begin(cmd_buff_begin_info);
  cmd_buffs[final_cmd_buffer_ind].bindPipeline(vk::PipelineBindPoint::eCompute,
                                               vulten_pipeline->pipeline);
  cmd_buffs[final_cmd_buffer_ind].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      vulten_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {descriptor_sets[final_cmd_buffer_ind]
           ->descriptor_set->vk_descriptor_set},  // List of descriptor sets
      {});                                        // Dynamic offsets

  adj_strides = vulten_utills::calculate_adj_strides(dims);
  reduce_shader::Push_const push_const = {
      uint32_t(dims[axis[final_cmd_buffer_ind]]),
      uint32_t(adj_strides[axis[final_cmd_buffer_ind]]),
      uint32_t(adj_strides[axis[final_cmd_buffer_ind] + 1]), op};
  cmd_buffs[final_cmd_buffer_ind].pushConstants(
      vulten_pipeline->pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
      sizeof(reduce_shader::Push_const), &push_const);

  total_elements /= push_const.axi_size;
  cmd_buffs[final_cmd_buffer_ind].dispatch(
      (total_elements /
       inst->device_propertys.props.limits.maxComputeWorkGroupInvocations) +
          1,
      1, 1);

  cmd_buffs[final_cmd_buffer_ind].end();

  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());

  vk::SubmitInfo SubmitInfo(
      axis_semaphores.size() > 0 ? 1 : 0,            // Num Wait Semaphores
      &axis_semaphores[axis_semaphores.size() - 1],  // Wait Semaphores
      wait_stages.data(),                            // Pipeline Stage Flags
      1,                                             // Num Command Buffers
      &cmd_buffs[final_cmd_buffer_ind]);             // List of command buffers
  queue_alloc.queue->vk_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  for (vk::Semaphore seamphore : axis_semaphores) {
    inst->logical_dev.destroySemaphore(seamphore);
  }
  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buffs);
  // inst->logical_dev.freeDescriptorSets(descriptor_pool, num_sets,
  //                                      descriptor_sets.data());
  // inst->logical_dev.destroyDescriptorPool(descriptor_pool);
}

}  // namespace reduce
}  // namespace vulten_ops