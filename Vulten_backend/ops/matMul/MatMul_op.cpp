#include "MatMul_op.h"

#include <cmath>
#include <memory>

#include "../transpose/Transpose_shader.h"
#include "MatMul_shader.h"

namespace vulten_ops {
namespace mat_mul {

void run_op(vulten_backend::Instance *inst, Data_type dt, Vulten_tensor a,
            bool trans_a, Mat_size mat_size_a, Vulten_tensor b, bool trans_b,
            Mat_size mat_size_b, Vulten_tensor output) {
  VULTEN_LOG_DEBUG("Running vulten_ops::MatMul_op<" + Data_type_to_str(dt) +
                   ">")
  vulten_backend::Queue_alloc queue_alloc =
      inst->get_queue(false, true, false, false);

  bool inline_transpose = false;
  if (output.get_total_elements() <= 32 * 32) {
    inline_transpose = true;
  }

  vulten_backend::Vulten_pipeline *transpose_pipeline = nullptr;
  if ((trans_a || trans_b) && !inline_transpose) {
    std::string transpose_pipe_string = "Transpose_" + Data_type_to_str(dt);
    transpose_pipeline = inst->get_cached_pipeline(transpose_pipe_string);
    if (transpose_pipeline == nullptr) {
      VULTEN_LOG_DEBUG("Creating vulten_ops::MatMul_op pipeline " +
                       transpose_pipe_string)

      transpose_shader::Spec_cons spec;
      spec.localX =
          inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;

      const std::vector<vk::SpecializationMapEntry> specs = {
          {0, offsetof(transpose_shader::Spec_cons, localX), sizeof(uint32_t)},
      };
      vk::SpecializationInfo spec_info(specs.size(), specs.data(), sizeof(spec),
                                       &spec);

      const std::vector<vk::PushConstantRange> push_const_ranges = {
          {vk::ShaderStageFlagBits::eCompute, 0,
           sizeof(transpose_shader::Push_const)},
      };

      transpose_shader::Generate_transpose_shader_info
          generate_transpose_shader_info{dt};
      transpose_pipeline =
          inst->create_pipeline(transpose_pipe_string,
                                std::vector<vk::DescriptorType>(
                                    2, vk::DescriptorType::eStorageBuffer),
                                transpose_shader::generate_transpose_shader(
                                    generate_transpose_shader_info),
                                &spec_info, push_const_ranges);
    } else {
      VULTEN_LOG_DEBUG("Using cached vulten_ops::MatMul_op pipeline " +
                       transpose_pipe_string)
    }
  }

  Mat_size mat_size_a_post_trans = {trans_a ? mat_size_a.y : mat_size_a.x,
                                    trans_a ? mat_size_a.x : mat_size_a.y};
  Mat_size mat_size_b_post_trans = {trans_b ? mat_size_b.y : mat_size_b.x,
                                    trans_b ? mat_size_b.x : mat_size_b.y};

  // Block size is a highly dependant on the input matrix and gpu
  // 4x4 is good on smaller gpus
  uint32_t max_block_size = 16;
  if (output.get_total_elements() <= 64 * 64) {
    max_block_size = 4;
  }
  uint32_t block_size_x = 0;
  for (int i = max_block_size; i > 0; i--) {
    if (mat_size_b_post_trans.x % i == 0 && mat_size_a_post_trans.x % i == 0) {
      block_size_x = i;
      break;
    }
  }
  uint32_t block_size_y = 0;
  for (int i = max_block_size; i > 0; i--) {
    if (mat_size_b_post_trans.y % i == 0 && mat_size_a_post_trans.y % i == 0) {
      block_size_y = i;
      break;
    }
  }

  uint32_t max_local_size =
      inst->device_propertys.props.limits.maxComputeWorkGroupInvocations;
  uint32_t local_x = 1;
  for (int i = max_local_size; i > 0; i--) {
    if (uint32_t(std::ceil(mat_size_a_post_trans.x / float(block_size_x))) %
            i ==
        0) {
      local_x = i;
      break;
    }
  }
  // This degraded perfromance idk...
  uint32_t local_y = 1;
  // for(int i = std::floor(max_local_size / local_x); i > 0; i--){
  //     if(uint32_t(std::ceil((trans_b ? mat_size_b.x : mat_size_b.y) /
  //     float(block_size_y))) % i == 0){
  //         local_y = i;
  //         break;
  //     }
  // }

  uint32_t bkNum = uint32_t(mat_size_a_post_trans.y / block_size_x);
  bool unroll_bk = false;
  // On some GPUs it helps others it hurts...
  // dissable for now
  // if(bkNum <= 4096)
  // unroll_bk = true;
  std::string matmul_pipe_string =
      "MatMul_" + Data_type_to_str(dt) + "_" + std::to_string(local_x) + "_" +
      std::to_string(local_y) + "_" + std::to_string(block_size_x) + "_" +
      std::to_string(block_size_y) + "_" + std::to_string(bkNum);
  if (inline_transpose) {
    matmul_pipe_string +=
        "_" + std::to_string(trans_a) + "_" + std::to_string(trans_b);
  }
  vulten_backend::Vulten_pipeline *matmul_pipeline =
      inst->get_cached_pipeline(matmul_pipe_string);
  if (matmul_pipeline == nullptr) {
    VULTEN_LOG_DEBUG("Creating vulten_ops::MatMul_op pipeline " +
                     matmul_pipe_string)

    mat_mul_shader::Spec_cons spec;
    spec.local_x = local_x;
    spec.local_y = local_y;
    spec.blockSizeX = block_size_x;
    spec.blockSizeY = block_size_y;
    spec.bkNum = bkNum;
    if (inline_transpose) {
      spec.transA = trans_a;
      spec.transB = trans_b;
    } else {
      spec.transA = false;
      spec.transB = false;
    }

    const std::vector<vk::SpecializationMapEntry> specs = {
        {0, 0, sizeof(uint32_t)},
        {1, offsetof(mat_mul_shader::Spec_cons, local_y), sizeof(uint32_t)},
        {2, offsetof(mat_mul_shader::Spec_cons, blockSizeX), sizeof(uint32_t)},
        {3, offsetof(mat_mul_shader::Spec_cons, blockSizeY), sizeof(uint32_t)},
        {4, offsetof(mat_mul_shader::Spec_cons, bkNum), sizeof(uint32_t)},
        {5, offsetof(mat_mul_shader::Spec_cons, transA), sizeof(VkBool32)},
        {6, offsetof(mat_mul_shader::Spec_cons, transB), sizeof(VkBool32)},
    };
    vk::SpecializationInfo spec_info(specs.size(), specs.data(),
                                     sizeof(mat_mul_shader::Spec_cons), &spec);

    const std::vector<vk::PushConstantRange> push_const_ranges = {
        {vk::ShaderStageFlagBits::eCompute, 0,
         sizeof(mat_mul_shader::Push_const)}};

    mat_mul_shader::Generate_matMul_shader_info generate_matMul_shader_info{
        dt, unroll_bk};
    std::vector<vk::DescriptorType> buffer_types =
        std::vector<vk::DescriptorType>(NUM_BUFFERS,
                                        vk::DescriptorType::eStorageBuffer);
    matmul_pipeline = inst->create_pipeline(
        matmul_pipe_string, buffer_types,
        mat_mul_shader::generate_matMul_shader(generate_matMul_shader_info),
        &spec_info, push_const_ranges);
  } else {
    VULTEN_LOG_DEBUG("Using cached vulten_ops::MatMul_op pipeline " +
                     matmul_pipe_string)
  }

  vulten_backend::Descriptor_set_alloc matmul_descriptor_set_alloc =
      inst->get_descriptor_sets(NUM_BUFFERS,
                                matmul_pipeline->descriptor_set_layout);
  vulten_backend::Descriptor_set_alloc transpose_a_descriptor_set_alloc =
      trans_a && !inline_transpose
          ? inst->get_descriptor_sets(2,
                                      transpose_pipeline->descriptor_set_layout)
          : vulten_backend::Descriptor_set_alloc();
  vulten_backend::Descriptor_set_alloc transpose_b_descriptor_set_alloc =
      trans_b && !inline_transpose
          ? inst->get_descriptor_sets(2,
                                      transpose_pipeline->descriptor_set_layout)
          : vulten_backend::Descriptor_set_alloc();

  vk::DescriptorSet transpose_a_descriptor_set;
  if (trans_a && !inline_transpose) {
    transpose_a_descriptor_set =
        transpose_a_descriptor_set_alloc.descriptor_set->vk_descriptor_set;
  }
  vk::DescriptorSet transpose_b_descriptor_set;
  if (trans_b && !inline_transpose) {
    transpose_b_descriptor_set =
        transpose_b_descriptor_set_alloc.descriptor_set->vk_descriptor_set;
  }
  vk::DescriptorSet matMul_descriptor_set =
      matmul_descriptor_set_alloc.descriptor_set->vk_descriptor_set;

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  std::vector<vk::CommandBuffer> cmd_buffs =
      inst->logical_dev.allocateCommandBuffers(cmd_buff_alloc_info);
  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
      std::vector<vk::WriteDescriptorSet>();

  std::unique_ptr<vulten_backend::Device_buffer> trans_a_buffer;
  vk::DescriptorBufferInfo transpose_a_buffer_info = vk::DescriptorBufferInfo();
  vk::DescriptorBufferInfo transpose_a_out_buffer_info =
      vk::DescriptorBufferInfo();
  if (trans_a && !inline_transpose) {
    trans_a_buffer = std::unique_ptr<vulten_backend::Device_buffer>(
        inst->create_device_buffer(a.buffer->buffer_size));

    transpose_a_buffer_info =
        vk::DescriptorBufferInfo(a.buffer->vk_buffer, 0, a.buffer->buffer_size);
    transpose_a_out_buffer_info = vk::DescriptorBufferInfo(
        trans_a_buffer.get()->vk_buffer, 0, a.buffer->buffer_size);

    writeDescriptorSets.push_back({transpose_a_descriptor_set, 0, 0, 1,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   &transpose_a_buffer_info});
    writeDescriptorSets.push_back({transpose_a_descriptor_set, 1, 0, 1,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   &transpose_a_out_buffer_info});
  }

  std::unique_ptr<vulten_backend::Device_buffer> trans_b_buffer;
  vk::DescriptorBufferInfo transpose_b_buffer_info = vk::DescriptorBufferInfo();
  vk::DescriptorBufferInfo transpose_b_out_buffer_info =
      vk::DescriptorBufferInfo();
  if (trans_b && !inline_transpose) {
    trans_b_buffer = std::unique_ptr<vulten_backend::Device_buffer>(
        inst->create_device_buffer(b.buffer->buffer_size));

    transpose_b_buffer_info =
        vk::DescriptorBufferInfo(b.buffer->vk_buffer, 0, b.buffer->buffer_size);
    transpose_b_out_buffer_info = vk::DescriptorBufferInfo(
        trans_b_buffer.get()->vk_buffer, 0, trans_b_buffer.get()->buffer_size);

    writeDescriptorSets.push_back({transpose_b_descriptor_set, 0, 0, 1,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   &transpose_b_buffer_info});
    writeDescriptorSets.push_back({transpose_b_descriptor_set, 1, 0, 1,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   &transpose_b_out_buffer_info});
  }

  vk::DescriptorBufferInfo a_buffer_info(trans_a && !inline_transpose
                                             ? trans_a_buffer.get()->vk_buffer
                                             : a.buffer->vk_buffer,
                                         0, a.buffer->buffer_size);
  vk::DescriptorBufferInfo b_buffer_info(trans_b && !inline_transpose
                                             ? trans_b_buffer.get()->vk_buffer
                                             : b.buffer->vk_buffer,
                                         0, b.buffer->buffer_size);
  vk::DescriptorBufferInfo output_buffer_info(output.buffer->vk_buffer, 0,
                                              output.buffer->buffer_size);

  writeDescriptorSets.push_back({matMul_descriptor_set, 0, 0, 1,
                                 vk::DescriptorType::eStorageBuffer, nullptr,
                                 &a_buffer_info});
  writeDescriptorSets.push_back({matMul_descriptor_set, 1, 0, 1,
                                 vk::DescriptorType::eStorageBuffer, nullptr,
                                 &b_buffer_info});
  writeDescriptorSets.push_back({matMul_descriptor_set, 2, 0, 1,
                                 vk::DescriptorType::eStorageBuffer, nullptr,
                                 &output_buffer_info});

  inst->logical_dev.updateDescriptorSets(writeDescriptorSets, {});

  int cmd_buff_indx = 0;

  cmd_buffs[cmd_buff_indx].begin(cmd_buff_begin_info);

  if (trans_a && !inline_transpose) {
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
    uint32_t threads = std::ceil(
        float(a.get_total_elements()) /
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
    cmd_buffs[cmd_buff_indx].dispatch(threads, 1, 1);
  }

  if (trans_b && !inline_transpose) {
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
    uint32_t threads = std::ceil(
        float(b.get_total_elements()) /
        inst->device_propertys.props.limits.maxComputeWorkGroupInvocations);
    cmd_buffs[cmd_buff_indx].dispatch(threads, 1, 1);
  }

  if ((trans_a && !inline_transpose) || (trans_b && !inline_transpose)) {
    vk::MemoryBarrier mem_bar = vk::MemoryBarrier(
        vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
        vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);
    cmd_buffs[cmd_buff_indx].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(),
        mem_bar, nullptr, nullptr);
  }

  cmd_buffs[cmd_buff_indx].bindPipeline(vk::PipelineBindPoint::eCompute,
                                        matmul_pipeline->pipeline);
  cmd_buffs[cmd_buff_indx].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      matmul_pipeline->pipeline_layout,  // Pipeline Layout
      0,                                 // First descriptor set
      {matMul_descriptor_set},           // List of descriptor sets
      {});

  cmd_buffs[cmd_buff_indx].fillBuffer(output.buffer->vk_buffer, 0,
                                      VK_WHOLE_SIZE, 0);
  vk::MemoryBarrier mem_bar = vk::MemoryBarrier(
      vk::AccessFlagBits::eTransferWrite,
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);
  cmd_buffs[cmd_buff_indx].pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), mem_bar,
      nullptr, nullptr);

  mat_mul_shader::Push_const pushes = {
      mat_size_a_post_trans.x, mat_size_a_post_trans.y, mat_size_b_post_trans.x,
      mat_size_b_post_trans.y};
  cmd_buffs[cmd_buff_indx].pushConstants(
      matmul_pipeline->pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
      sizeof(mat_mul_shader::Push_const), &pushes);
  uint32_t bkNumX = std::ceil(
      std::ceil(mat_size_a_post_trans.x / float(block_size_x)) / local_x);
  uint32_t bkNumY = std::ceil(
      std::ceil(mat_size_b_post_trans.y / float(block_size_y)) / local_y);
  cmd_buffs[cmd_buff_indx].dispatch(bkNumX, bkNumY, 1);
  cmd_buffs[cmd_buff_indx].end();

  vk::Fence fence = inst->logical_dev.createFence(vk::FenceCreateInfo());
  vk::SubmitInfo SubmitInfo(
      0,                           // Num Wait Semaphores
      nullptr,                     // Wait Semaphores
      nullptr,                     // Pipeline Stage Flags
      1,                           // Num Command Buffers
      &cmd_buffs[cmd_buff_indx]);  // List of command buffers
  queue_alloc.queue->vk_queue.submit({SubmitInfo}, fence);
  vk::Result fenceRes =
      inst->logical_dev.waitForFences({fence},        // List of fences
                                      true,           // Wait All
                                      uint64_t(-1));  // Timeout

  inst->logical_dev.destroyFence(fence);
  inst->logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buffs);
}

}  // namespace mat_mul
}  // namespace vulten_ops