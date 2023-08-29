#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "VulkanMemoryAllocator/include/vk_mem_alloc.h"
#include "vulten_logger.h"

#define VOID_TO_INSTANCE(X) static_cast<vulten_backend::Instance *>(X)
#define VOID_TO_DEVICE_BUFFER(X) static_cast<vulten_backend::Device_buffer *>(X)
#define VOID_TO_HOST_MAPPABLE_BUFFER(X) \
  static_cast<vulten_backend::Host_mappable_buffer *>(X)
#define CALL_ALL_BASIC_TYPES(func)                                            \
  func(TF_FLOAT) func(TF_HALF) func(TF_DOUBLE) func(TF_INT32) func(TF_UINT32) \
      func(TF_INT8) func(TF_UINT8) func(TF_INT64) func(TF_UINT64)             \
          func(TF_INT16) func(TF_UINT16)
#define CALL_COMPLEX(func) func(TF_COMPLEX64) func(TF_COMPLEX128)
#define CALL_ALL_TYPES(func) CALL_ALL_BASIC_TYPES(func) CALL_COMPLEX(func)

namespace vulten_backend {

struct Device_queue_prop {
  uint32_t max_queues;
  bool hasCompute, hasTransfer, hasGraphics;
};

struct Device_property {
  vk::PhysicalDeviceProperties props;
  vk::PhysicalDeviceProperties2 props2;
  std::vector<Device_queue_prop> queue_props;
  vk::PhysicalDeviceMemoryProperties mem_props;
  std::vector<std::string> extens;
  uint32_t subgroupSize;
};

struct Device_propertys {
 private:
 public:
  std::vector<vulten_backend::Device_property> *devices;

  Device_propertys();
  ~Device_propertys();
};

struct Vk_instance {
 private:
 public:
  vk::Instance *instance;

  Vk_instance();
  ~Vk_instance();
};

class Instance;
struct alignas(64) Buffer {
 private:
 public:
  Instance *inst;
  vk::Buffer vk_buffer;
  VmaAllocationInfo allocInfo;
  VmaAllocation allocation;

  uint32_t buffer_size;

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties,
                          vk::PhysicalDeviceMemoryProperties &memProperties);

  Buffer(Instance *inst, uint32_t size, bool trans_src, bool trans_dst);
  ~Buffer();
};

struct alignas(64) Host_mappable_buffer : Buffer {
 private:
  //
 public:
  //

  Host_mappable_buffer(Instance *instance, void *data, uint32_t size,
                       bool sync_to_device, bool trans_src, bool trans_dst,
                       bool staging);
  ~Host_mappable_buffer();
};

struct alignas(64) Device_buffer : Buffer {
 private:
  //
 public:
  Device_buffer(Instance *instance, uint32_t size, bool trans_src,
                bool trans_dst);
  ~Device_buffer();
};

struct Vulten_pipeline {
 private:
  //
 public:
  bool auto_clean;
  vulten_backend::Instance *inst;
  vk::Pipeline pipeline;
  vk::PipelineLayout pipeline_layout;
  vk::ShaderModule shader;
  vk::DescriptorSetLayout descriptor_set_layout;
  vk::PipelineCache pipeline_cache;

  /**
   * @param instance reference to vulten_backend::Instance
   * @param num_buffers Number of buffers needed for op.
   * @param shader_source Source spv for shader.
   * @param specs Vector of spec contrantes.
   */
  Vulten_pipeline(vulten_backend::Instance *instance, uint32_t num_buffers,
                  const std::vector<uint32_t> &shader_source,
                  vk::SpecializationInfo *spec_info = {},
                  std::vector<vk::PushConstantRange> push_ranges = {});
  Vulten_pipeline();
  ~Vulten_pipeline();
};

class alignas(64) Instance {
 private:
  //
 public:
  uint32_t device_num;
  Device_property device_propertys;
  vk::PhysicalDevice physical_dev;
  vk::Device logical_dev;
  std::mutex main_queue_mutex;
  vk::Queue main_queue;
  vk::CommandPool cmd_pool;
  VmaAllocator allocator;
  mutable std::shared_timed_mutex pipe_mutex;
  std::unordered_map<std::string, Vulten_pipeline *> pipelines;

  Host_mappable_buffer *create_host_mappable_buffer(void *data, uint32_t size,
                                                    bool sync_to_device = true,
                                                    bool trans_src = true,
                                                    bool trans_dst = true,
                                                    bool staging = false);
  Device_buffer *create_device_buffer(uint32_t size, bool trans_src = true,
                                      bool trans_dst = true);
  void copy_buffer(Buffer *src, Buffer *dest, bool lock = true,
                   uint32_t size = 0);
  void fill_buffer(Buffer *dstBuffer, uint64_t offset, uint64_t size,
                   uint32_t data, bool lock = true);
  Vulten_pipeline *get_cached_pipeline(std::string pipe_string);
  Vulten_pipeline *create_pipeline(
      std::string pipe_string, uint32_t num_buffers,
      std::vector<uint32_t> shader_spv, vk::SpecializationInfo *spec_info = {},
      std::vector<vk::PushConstantRange> push_ranges = {});

  // Instance(const Instance&) = delete;
  Instance(uint32_t dev_num);
  ~Instance();
};

}  // namespace vulten_backend