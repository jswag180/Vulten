#pragma once

#include <unordered_map>

#include "../Vulten_backend.h"

namespace vulten_ops {

enum Data_type {
  VULTEN_FLOAT = 0,
  VULTEN_FLOAT16 = 1,
  VULTEN_DOUBLE = 2,
  VULTEN_INT32 = 3,
  VULTEN_UINT32 = 4,
  VULTEN_INT64 = 5,
  VULTEN_UINT64 = 6,
  VULTEN_INT8 = 7,
  VULTEN_UINT8 = 8,
};

std::string Data_type_to_str(Data_type dt);

struct Vulten_tensor {
  vulten_backend::Buffer *buffer;
  uint32_t *dims;
  uint32_t num_dims;

  uint32_t get_total_elements();

  Vulten_tensor(vulten_backend::Buffer *buffer_ptr, uint32_t num_dims,
                uint32_t *dims_ptr);
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
                  vk::SpecializationInfo spec_info = {});
  Vulten_pipeline();
  ~Vulten_pipeline();
};

class Vulten_op {
 private:
  //
 public:
  vulten_backend::Instance *inst;
  Data_type data_type;
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
      vk::SpecializationInfo spec_info = {});

  Vulten_op(vulten_backend::Instance &inst, Data_type dt);
  ~Vulten_op();
};

}  // namespace vulten_ops