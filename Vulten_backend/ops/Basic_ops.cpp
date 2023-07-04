#include "Basic_ops.h"

#include <cmath>

#include "../../shaders/headers/BasicOps/BasicOps.comp.h"
#include "Vulten_backend/Vulten_backend.h"
#include "Vulten_backend/Vulten_utills.h"

#define NUM_BUFFERS 5
#define NUM_SETS 1

namespace vulten_ops {

struct Push_const {
  uint32_t batch_num;
};

std::string op_as_str(uint32_t op) {
  if (op == OP_MUL) {
    return "Mul";
  } else if (op == OP_ADD) {
    return "Add";
  } else if (op == OP_SUB) {
    return "Sub";
  } else if (op == OP_DIV) {
    return "Div";
  } else if (op == OP_DIV_NO_NAN) {
    return "DivNoNan";
  } else {
    return "INVALID";
  }
}

Basic_op::Basic_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::Basic_op")
}

void Basic_op::run_op(Data_type dt, uint32_t op, Vulten_tensor x,
                      Vulten_tensor y, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::Basic_op<" + Data_type_to_str(dt) +
                   ", " + op_as_str(op) + ">")
  inst->main_queue_mutex.lock();

  std::string basic_pipe_string =
      "Basic_" + op_as_str(op) + "_" + Data_type_to_str(dt);
  Vulten_pipeline *vulten_pipeline = nullptr;
  if (!is_pipeline_cached(basic_pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::Basic_op<" + Data_type_to_str(dt) +
                     ", " + op_as_str(op) + "> " + basic_pipe_string)

    struct Spec {
      uint32_t local_x;

      uint32_t op;
    } spec;

    spec.local_x =
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

    spec.op = op;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(uint32_t)}, {1, offsetof(Spec, op), sizeof(uint32_t)}};
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 3},
    };

    std::vector<Data_type> type_chain = {dt};
    vulten_pipeline = create_pipeline(
        basic_pipe_string, NUM_BUFFERS, BasicOps_comp, type_chain.data(),
        type_chain.size(), &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::Basic_op<" +
                     Data_type_to_str(dt) + ", " + op_as_str(op) + "> " +
                     basic_pipe_string)
    vulten_pipeline = pipelines[basic_pipe_string];
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
      descriptor_pool, NUM_SETS, &vulten_pipeline->descriptor_set_layout);
  vk::DescriptorSet descriptor_set =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info)
          .front();

  vk::DescriptorBufferInfo x_buffer_info(x.buffer->vk_buffer, 0,
                                         x.buffer->buffer_size);
  vk::DescriptorBufferInfo y_buffer_info(y.buffer->vk_buffer, 0,
                                         y.buffer->buffer_size);

  std::vector<uint32_t> adj_strides =
      vulten_utills::calculate_adj_strides(x.dims, x.num_dims);
  std::vector<uint32_t> y_strides =
      vulten_utills::calculate_adj_strides(y.dims, y.num_dims);
  adj_strides.insert(adj_strides.end(), y_strides.begin(), y_strides.end());
  std::vector<uint32_t> output_strides =
      vulten_utills::calculate_adj_strides(output.dims, output.num_dims);
  adj_strides.insert(adj_strides.end(), output_strides.begin(),
                     output_strides.end());
  auto strides_stageing = std::unique_ptr<vulten_backend::Host_mappable_buffer>(
      inst->create_host_mappable_buffer((uint8_t *)adj_strides.data(),
                                        sizeof(uint32_t) * adj_strides.size()));
  auto strides = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(strides_stageing->buffer_size, false, true));
  inst->copy_buffer(strides_stageing.get(), strides.get(), false);
  vk::DescriptorBufferInfo strides_buffer_info(strides->vk_buffer, 0,
                                               strides->buffer_size);

  std::vector<uint32_t> dims_vec = std::vector<uint32_t>();
  for (uint32_t i = 0; i < x.num_dims; i++) {
    dims_vec.push_back(x.dims[i]);
  }
  for (uint32_t i = 0; i < y.num_dims; i++) {
    dims_vec.push_back(y.dims[i]);
  }
  auto dims_stageing = std::unique_ptr<vulten_backend::Host_mappable_buffer>(
      inst->create_host_mappable_buffer((uint8_t *)dims_vec.data(),
                                        sizeof(uint32_t) * dims_vec.size()));
  auto dims = std::unique_ptr<vulten_backend::Device_buffer>(
      inst->create_device_buffer(dims_stageing->buffer_size, false, true));
  inst->copy_buffer(dims_stageing.get(), dims.get(), false);
  vk::DescriptorBufferInfo dims_buffer_info(dims->vk_buffer, 0,
                                            dims->buffer_size);

  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);
  const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &x_buffer_info},
      {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &y_buffer_info},
      {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &strides_buffer_info},
      {descriptor_set, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &dims_buffer_info},
      {descriptor_set, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
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
      {descriptor_set},                  // List of descriptor sets
      {});                               // Dynamic offsets

  uint32_t pushes[3] = {uint32_t(x.num_dims), uint32_t(y.num_dims),
                        uint32_t(output.num_dims)};
  cmd_buffs.pushConstants(vulten_pipeline->pipeline_layout,
                          vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(uint32_t) * 3, pushes);
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

Basic_op::~Basic_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::Basic_op") }

}  // namespace vulten_ops
