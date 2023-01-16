#pragma once

#include <unordered_map>

#include "../Vulten_backend.h"

namespace vulten_ops {

enum Data_type {
  VULTEN_FLOAT = 1,
  VULTEN_FLOAT16 = 19,
  VULTEN_DOUBLE = 2,
  VULTEN_INT32 = 3,
  VULTEN_UINT32 = 22,
  VULTEN_INT64 = 9,
  VULTEN_UINT64 = 23,
  VULTEN_INT8 = 6,
  VULTEN_UINT8 = 4,
};

#define VULTEN_DEFINE_BASIC_TYPES(op) \
  op(VULTEN_FLOAT)    \
  op(VULTEN_FLOAT16)  \
  op(VULTEN_DOUBLE)   \
  op(VULTEN_INT32)    \
  op(VULTEN_UINT32)   \
  op(VULTEN_INT8)     \
  op(VULTEN_UINT8)    \
  op(VULTEN_INT64)    \
  op(VULTEN_UINT64)

std::string Data_type_to_str(Data_type dt);

struct Vulten_tensor {
  vulten_backend::Buffer *buffer;
  int64_t *dims;
  int64_t num_dims;

  int64_t get_total_elements();

  Vulten_tensor(vulten_backend::Buffer *buffer_ptr, int64_t num_dims,
                int64_t *dims_ptr);
  ~Vulten_tensor();
};

struct Vulten_pipeline {
 private:
  //
 public:
  bool auto_clean;
  vulten_backend::Instance *inst;
  vk::Pipeline pipeline;
  vk::DescriptorSet descriptor_set;
  vk::PipelineLayout pipeline_layout;
  vk::ShaderModule shader;
  vk::DescriptorSetLayout descriptor_set_layout;
  vk::PipelineCache pipeline_cache;
  vk::DescriptorPool descriptor_pool;

  /**
   * @param instance reference to vulten_backend::Instance
   * @param num_buffers Number of buffers needed for op.
   * @param shader_source Source spv for shader.
   * @param specs Vector of spec contrantes.
   */
  Vulten_pipeline(vulten_backend::Instance &instance, uint32_t num_buffers,
                  const std::vector<uint32_t> &shader_source,
                  vk::SpecializationInfo *spec_info = {});
  Vulten_pipeline();
  ~Vulten_pipeline();
};

class Vulten_op {
 private:
  //
 public:
  vulten_backend::Instance *inst;
  std::unordered_map<std::string, Vulten_pipeline *> pipelines;

  virtual void run_op();
  /**
   * See if pipeline is cached.
   * @param pipe_string A string of format shaderName_dType_spec1_spec_2_...
   * @return If the pipeline is cached.
   */
  bool is_pipeline_cached(std::string pipe_string);
  /**
   * Will create pipeline and cache it.
   *
   * @param pipe_string A string of format shaderName_dType_spec1_spec_2_...
   * @return requested pipeline.
   */
  virtual Vulten_pipeline *create_pipeline(
      std::string pipe_string, uint32_t num_buffers,
      const std::vector<uint32_t> &shader_source,
      vk::SpecializationInfo *spec_info = {});

  Vulten_op(vulten_backend::Instance *inst);
  ~Vulten_op();
};

}  // namespace vulten_ops