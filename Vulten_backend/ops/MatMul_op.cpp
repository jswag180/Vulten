#include "MatMul_op.h"

#include <memory>

#include "../../shaders/headers/MatMul/MatMul.comp.h"
#include "../../shaders/headers/Transpose/Transpose.comp.h"

namespace vulten_ops {

MatMul_op::MatMul_op(vulten_backend::Instance *inst) : Vulten_op(inst) {
  VULTEN_LOG_DEBUG("Creating vulten_ops::MatMul_op")
}

void MatMul_op::run_op(Data_type dt, Vulten_tensor a, bool trans_a,
                       Mat_size mat_size_a, Vulten_tensor b, bool trans_b,
                       Mat_size mat_size_b, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::MatMul_op<" + Data_type_to_str(dt) +
                   ">")
  inst->main_queue_mutex.lock();

  Vulten_pipeline *transpose_pipeline = nullptr;
  if (trans_a || trans_b) {
    std::string transpose_pipe_string = "Transpose_" + Data_type_to_str(dt);
    if (!is_pipeline_cached(transpose_pipe_string)) {
      VULTEN_LOG_DEBUG("Creating vulten_ops::MatMul_op pipeline " +
                       transpose_pipe_string)

      const std::vector<vk::PushConstantRange> push_const_ranges = {
          {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2},
      };

      std::vector<Data_type> type_chain = {dt};
      transpose_pipeline = create_pipeline(
          transpose_pipe_string, 2, Transpose_comp, type_chain.data(),
          type_chain.size(), nullptr, push_const_ranges);
    } else {
      VULTEN_LOG_DEBUG("Using cached vulten_ops::MatMul_op pipeline " +
                       transpose_pipe_string)
      transpose_pipeline = pipelines[transpose_pipe_string];
    }
  }

  std::string matmul_pipe_string =
      "MatMul_" + Data_type_to_str(dt) + "_" + std::to_string(mat_size_a.x) +
      "_" + std::to_string(mat_size_a.y) + "_" + std::to_string(mat_size_b.x) +
      "_" + std::to_string(mat_size_b.y);
  Vulten_pipeline *matmul_pipeline = nullptr;
  if (!is_pipeline_cached(matmul_pipe_string)) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::MatMul_op pipeline " +
                     matmul_pipe_string)

    struct Spec {
      uint32_t ax, ay;
      uint32_t bx, by;
    } spec;

    spec.ax = trans_a ? mat_size_a.y : mat_size_a.x;
    spec.ay = trans_a ? mat_size_a.x : mat_size_a.y;

    spec.bx = trans_b ? mat_size_b.y : mat_size_b.x;
    spec.by = trans_b ? mat_size_b.x : mat_size_b.y;

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, offsetof(Spec, ax), sizeof(uint32_t)},
        {1, offsetof(Spec, ay), sizeof(uint32_t)},
        {2, offsetof(Spec, bx), sizeof(uint32_t)},
        {3, offsetof(Spec, by), sizeof(uint32_t)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                     &spec);

    std::vector<Data_type> type_chain = {dt};
    matmul_pipeline =
        create_pipeline(matmul_pipe_string, 3, MatMul_comp, type_chain.data(),
                        type_chain.size(), &spec_info);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::MatMul_op pipeline " +
                     matmul_pipe_string)
    matmul_pipeline = pipelines[matmul_pipe_string];
  }

  vk::DescriptorPool descriptor_pool;
  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer,
      2 + (trans_a ? 2 : 0) + (trans_b ? 2 : 0));
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      1 + (trans_a ? 1 : 0) + (trans_b ? 1 : 0), descriptor_pool_size);
  descriptor_pool =
      inst->logical_dev.createDescriptorPool(descriptor_pool_create_info);

  std::vector<vk::DescriptorSetLayout> descriptor_set_layouts =
      std::vector<vk::DescriptorSetLayout>(1 + (trans_a ? 1 : 0) +
                                           (trans_b ? 1 : 0));
  if (trans_a)
    descriptor_set_layouts[0] = transpose_pipeline->descriptor_set_layout;
  if (trans_b)
    descriptor_set_layouts[trans_a ? 1 : 0] =
        transpose_pipeline->descriptor_set_layout;
  descriptor_set_layouts[descriptor_set_layouts.size() - 1] =
      matmul_pipeline->descriptor_set_layout;

  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(
      descriptor_pool, descriptor_set_layouts.size(),
      descriptor_set_layouts.data());
  std::vector<vk::DescriptorSet> descriptor_sets =
      inst->logical_dev.allocateDescriptorSets(descriptor_set_alloc_info);

  vk::DescriptorSet transpose_a_descriptor_set;
  if (trans_a) transpose_a_descriptor_set = descriptor_sets[0];
  vk::DescriptorSet transpose_b_descriptor_set;
  if (trans_b) transpose_b_descriptor_set = descriptor_sets[trans_a ? 1 : 0];
  vk::DescriptorSet matMul_descriptor_set =
      descriptor_sets[descriptor_set_layouts.size() - 1];

  std::vector<vk::Semaphore> transpose_semaphores =
      std::vector<vk::Semaphore>();
  std::vector<vk::PipelineStageFlags> wait_stages =
      std::vector<vk::PipelineStageFlags>();

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      inst->cmd_pool, vk::CommandBufferLevel::ePrimary,
      1 + (trans_a ? 1 : 0) + (trans_b ? 1 : 0));
  std::vector<vk::CommandBuffer> cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);
  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  std::unique_ptr<vulten_backend::Device_buffer> trans_a_buffer;
  if (trans_a) {
    transpose_semaphores.push_back(
        inst->logical_dev.createSemaphore(vk::SemaphoreCreateInfo()));
    wait_stages.push_back(vk::PipelineStageFlagBits::eComputeShader);
    trans_a_buffer = std::unique_ptr<vulten_backend::Device_buffer>(
        inst->create_device_buffer(a.buffer->buffer_size));

    vk::DescriptorBufferInfo transpose_a_buffer_info(a.buffer->vk_buffer, 0,
                                                     a.buffer->buffer_size);
    vk::DescriptorBufferInfo transpose_a_out_buffer_info(
        trans_a_buffer.get()->vk_buffer, 0, a.buffer->buffer_size);

    std::vector<vk::WriteDescriptorSet> transpose_a_WriteDescriptorSets =
        std::vector<vk::WriteDescriptorSet>(2);
    transpose_a_WriteDescriptorSets[0] = {transpose_a_descriptor_set,
                                          0,
                                          0,
                                          1,
                                          vk::DescriptorType::eStorageBuffer,
                                          nullptr,
                                          &transpose_a_buffer_info};
    transpose_a_WriteDescriptorSets[1] = {transpose_a_descriptor_set,
                                          1,
                                          0,
                                          1,
                                          vk::DescriptorType::eStorageBuffer,
                                          nullptr,
                                          &transpose_a_out_buffer_info};
    inst->logical_dev.updateDescriptorSets(transpose_a_WriteDescriptorSets, {});

    int cmd_buff_indx = 0;

    cmd_buffs[cmd_buff_indx].begin(cmd_buff_begin_info);
    cmd_buffs[cmd_buff_indx].bindPipeline(vk::PipelineBindPoint::eCompute,
                                          transpose_pipeline->pipeline);
    cmd_buffs[cmd_buff_indx].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,      // Bind point
        transpose_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                    // First descriptor set
        {transpose_a_descriptor_set},         // List of descriptor sets
        {});

    cmd_buffs[cmd_buff_indx].pushConstants(transpose_pipeline->pipeline_layout,
                                           vk::ShaderStageFlagBits::eCompute, 0,
                                           sizeof(mat_size_a), &mat_size_a);

    cmd_buffs[cmd_buff_indx].dispatch(a.get_total_elements(), 1, 1);
    cmd_buffs[cmd_buff_indx].end();

    vk::SubmitInfo SubmitInfo(
        0,                                      // Num Wait Semaphores
        nullptr,                                // Wait Semaphores
        nullptr,                                // Pipeline Stage Flags
        1,                                      // Num Command Buffers
        &cmd_buffs[cmd_buff_indx],              // List of command buffers
        1,                                      // Num Signal Semaphores
        &transpose_semaphores[cmd_buff_indx]);  // Signal Semaphores
    inst->main_queue.submit({SubmitInfo});
  }

  std::unique_ptr<vulten_backend::Device_buffer> trans_b_buffer;
  if (trans_b) {
    transpose_semaphores.push_back(
        inst->logical_dev.createSemaphore(vk::SemaphoreCreateInfo()));
    wait_stages.push_back(vk::PipelineStageFlagBits::eComputeShader);
    trans_b_buffer = std::unique_ptr<vulten_backend::Device_buffer>(
        inst->create_device_buffer(b.buffer->buffer_size));

    vk::DescriptorBufferInfo transpose_b_buffer_info(b.buffer->vk_buffer, 0,
                                                     b.buffer->buffer_size);
    vk::DescriptorBufferInfo transpose_b_out_buffer_info(
        trans_b_buffer.get()->vk_buffer, 0, trans_b_buffer.get()->buffer_size);

    std::vector<vk::WriteDescriptorSet> transpose_b_WriteDescriptorSets =
        std::vector<vk::WriteDescriptorSet>(2);
    transpose_b_WriteDescriptorSets[0] = {transpose_b_descriptor_set,
                                          0,
                                          0,
                                          1,
                                          vk::DescriptorType::eStorageBuffer,
                                          nullptr,
                                          &transpose_b_buffer_info};
    transpose_b_WriteDescriptorSets[1] = {transpose_b_descriptor_set,
                                          1,
                                          0,
                                          1,
                                          vk::DescriptorType::eStorageBuffer,
                                          nullptr,
                                          &transpose_b_out_buffer_info};
    inst->logical_dev.updateDescriptorSets(transpose_b_WriteDescriptorSets, {});

    int cmd_buff_indx = trans_a ? 1 : 0;

    cmd_buffs[cmd_buff_indx].begin(cmd_buff_begin_info);
    cmd_buffs[cmd_buff_indx].bindPipeline(vk::PipelineBindPoint::eCompute,
                                          transpose_pipeline->pipeline);
    cmd_buffs[cmd_buff_indx].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,      // Bind point
        transpose_pipeline->pipeline_layout,  // Pipeline Layout
        0,                                    // First descriptor set
        {transpose_b_descriptor_set},         // List of descriptor sets
        {});

    cmd_buffs[cmd_buff_indx].pushConstants(transpose_pipeline->pipeline_layout,
                                           vk::ShaderStageFlagBits::eCompute, 0,
                                           sizeof(mat_size_b), &mat_size_b);

    cmd_buffs[cmd_buff_indx].dispatch(b.get_total_elements(), 1, 1);
    cmd_buffs[cmd_buff_indx].end();

    vk::SubmitInfo SubmitInfo(
        0,                                      // Num Wait Semaphores
        nullptr,                                // Wait Semaphores
        nullptr,                                // Pipeline Stage Flags
        1,                                      // Num Command Buffers
        &cmd_buffs[cmd_buff_indx],              // List of command buffers
        1,                                      // Num Signal Semaphores
        &transpose_semaphores[cmd_buff_indx]);  // Signal Semaphores
    inst->main_queue.submit({SubmitInfo});
  }

  vk::DescriptorBufferInfo a_buffer_info(
      trans_a ? trans_a_buffer.get()->vk_buffer : a.buffer->vk_buffer, 0,
      a.buffer->buffer_size);
  vk::DescriptorBufferInfo b_buffer_info(
      trans_b ? trans_b_buffer.get()->vk_buffer : b.buffer->vk_buffer, 0,
      b.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);
  std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
      {matMul_descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer,
       nullptr, &a_buffer_info},
      {matMul_descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer,
       nullptr, &b_buffer_info},
      {matMul_descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer,
       nullptr, &output_buffer_info},
  };
  inst->logical_dev.updateDescriptorSets(WriteDescriptorSets, {});

  int cmd_buff_indx = cmd_buffs.size() - 1;

  cmd_buffs[cmd_buff_indx].begin(cmd_buff_begin_info);
  cmd_buffs[cmd_buff_indx].bindPipeline(vk::PipelineBindPoint::eCompute,
                                        matmul_pipeline->pipeline);
  cmd_buffs[cmd_buff_indx].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      matmul_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {matMul_descriptor_set},           // List of descriptor sets
      {});

  cmd_buffs[cmd_buff_indx].dispatch(output.dims[0], output.dims[1], 1);
  cmd_buffs[cmd_buff_indx].end();

  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());
  vk::SubmitInfo SubmitInfo(
      transpose_semaphores.size(),  // Num Wait Semaphores
      transpose_semaphores.data(),  // Wait Semaphores
      wait_stages.data(),           // Pipeline Stage Flags
      1,                            // Num Command Buffers
      &cmd_buffs[cmd_buff_indx]);   // List of command buffers
  inst->main_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  for (vk::Semaphore seamphore : transpose_semaphores) {
    inst->logical_dev.destroySemaphore(seamphore);
  }
  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(inst->cmd_pool, cmd_buffs);
  for (vk::DescriptorSet descriptor_set : descriptor_sets) {
    inst->logical_dev.freeDescriptorSets(descriptor_pool, 1, &descriptor_set);
  }
  inst->logical_dev.destroyDescriptorPool(descriptor_pool);
  inst->main_queue_mutex.unlock();
}

MatMul_op::~MatMul_op() { VULTEN_LOG_DEBUG("Freeing vulten_ops::MatMul_op") }

}  // namespace vulten_ops