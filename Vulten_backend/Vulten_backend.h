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
#include "Vulten_backend/Vulten_utills.h"

#define VULTEN_DISABLE_FLOAT16 "VULTEN_DISABLE_FLOAT16"
#define VULTEN_DISABLE_FLOAT64 "VULTEN_DISABLE_FLOAT64"
#define VULTEN_DISABLE_INT8 "VULTEN_DISABLE_INT8"
#define VULTEN_DISABLE_INT16 "VULTEN_DISABLE_INT16"
#define VULTEN_DISABLE_INT64 "VULTEN_DISABLE_INT64"

#define VOID_TO_INSTANCE(X) static_cast<vulten_backend::Instance *>(X)
#define VOID_TO_DEVICE_BUFFER(X) static_cast<vulten_backend::Device_buffer *>(X)
#define VOID_TO_HOST_MAPPABLE_BUFFER(X) \
  static_cast<vulten_backend::Host_mappable_buffer *>(X)
#define CALL_HALF(func)                                                  \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_FLOAT16)) { \
    func(TF_HALF)                                                        \
  }
#define CALL_DOUBLE(func)                                                \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_FLOAT64)) { \
    func(TF_DOUBLE)                                                      \
  }
#define CALL_INT64(func)                                               \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT64)) { \
    func(TF_INT64)                                                     \
  }
#define CALL_BOOL(func)                                               \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT8)) { \
    func(TF_BOOL)                                                     \
  }
#define CALL_ALL_BASIC_TYPES(func)                                       \
  func(TF_FLOAT) func(TF_INT32)                                          \
      func(TF_UINT32) if (!vulten_utills::get_env_bool(       \
                              VULTEN_DISABLE_FLOAT16)) {                 \
    func(TF_HALF)                                                        \
  }                                                                      \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT16)) {   \
    func(TF_INT16) func(TF_UINT16)                                       \
  }                                                                      \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT8)) {    \
    func(TF_INT8) func(TF_UINT8)                                         \
  }                                                                      \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_FLOAT64)) { \
    func(TF_DOUBLE)                                                      \
  }                                                                      \
  if (!vulten_utills::get_env_bool(VULTEN_DISABLE_INT64)) {   \
    func(TF_INT64) func(TF_UINT64)                                       \
  }
#define CALL_COMPLEX(func)                                        \
  func(TF_COMPLEX64) if (!vulten_utills::get_env_bool( \
                             VULTEN_DISABLE_FLOAT64)) {           \
    func(TF_COMPLEX128)                                           \
  }
#define CALL_ALL_TYPES(func) CALL_ALL_BASIC_TYPES(func) CALL_COMPLEX(func)

namespace vulten_backend {

struct Device_queue_prop {
  uint32_t max_queues;
  bool hasCompute, hasTransfer, hasGraphics, hasSparse;
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
                       bool staging, bool uniform);
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

  /**
   * @param instance reference to vulten_backend::Instance
   * @param num_buffers Number of buffers needed for op.
   * @param shader_source Source spv for shader.
   * @param specs Vector of spec contrantes.
   */
  Vulten_pipeline(vulten_backend::Instance *instance,
                  std::vector<vk::DescriptorType> &buffer_types,
                  const std::vector<uint32_t> &shader_source,
                  vk::SpecializationInfo *spec_info = {},
                  std::vector<vk::PushConstantRange> push_ranges = {});
  Vulten_pipeline();
  ~Vulten_pipeline();
};

struct Queue {
  std::mutex queue_mutex;
  vk::Queue vk_queue;
  vk::CommandPool cmd_pool;
  bool graphics, compute, transfer, sparse;

  Queue(){};
};

struct Queue_alloc {
  Queue *queue;

  Queue_alloc(const Queue_alloc &) = delete;
  Queue_alloc(Queue_alloc &&out) noexcept : queue(std::move(out.queue)) {}
  Queue_alloc(Queue *queue) : queue(queue){};
  ~Queue_alloc() { queue->queue_mutex.unlock(); };
};

struct Descriptor_set {
  std::shared_ptr<std::mutex> mutex;
  int descriptors;
  vk::DescriptorSet vk_descriptor_set;

  Descriptor_set() { mutex = std::shared_ptr<std::mutex>(new std::mutex); };
};

struct Descriptor_set_alloc {
  std::shared_ptr<Descriptor_set> descriptor_set;

  // Descriptor_set_alloc& operator=(Descriptor_set_alloc other){
  //     //std::swap(descriptor_set, other.descriptor_set);
  //     descriptor_set = other.descriptor_set;
  //     other.descriptor_set = nullptr;
  //     return *this;
  // }
  Descriptor_set_alloc(const Descriptor_set_alloc &) = delete;
  Descriptor_set_alloc(Descriptor_set_alloc &&out) noexcept
      : descriptor_set(std::move(out.descriptor_set)) {}
  Descriptor_set_alloc(std::shared_ptr<Descriptor_set> descriptor_set)
      : descriptor_set(descriptor_set){};
  Descriptor_set_alloc(){};
  ~Descriptor_set_alloc() {
    if (descriptor_set.get() != nullptr) descriptor_set->mutex->unlock();
  };
};

class alignas(64) Instance {
 private:
  //
 public:
  uint32_t device_num;
  Device_property device_propertys;
  vk::PhysicalDevice physical_dev;
  vk::Device logical_dev;
  vk::PipelineCache pipeline_cache;
  int total_queues;
  Queue *queues;
  VmaAllocator allocator;
  mutable std::shared_timed_mutex pipe_mutex;
  std::unordered_map<std::string, Vulten_pipeline *> pipelines;

  vk::DescriptorPool descriptor_pool;
  mutable std::shared_timed_mutex descriptor_mutex;
  std::vector<std::shared_ptr<Descriptor_set>> descriptors;

  Host_mappable_buffer *create_host_mappable_buffer(void *data, uint32_t size,
                                                    bool sync_to_device = true,
                                                    bool trans_src = true,
                                                    bool trans_dst = true,
                                                    bool staging = false,
                                                    bool uniform = false);
  Device_buffer *create_device_buffer(uint32_t size, bool trans_src = true,
                                      bool trans_dst = true);
  void copy_buffer(Buffer *src, Buffer *dest, uint32_t size = 0);
  void copy_buffer(Queue_alloc *queue_alloc, Buffer *src, Buffer *dest,
                   uint32_t size = 0);
  void fill_buffer(Buffer *dstBuffer, uint64_t offset, uint64_t size,
                   uint32_t data);
  void fill_buffer(Queue_alloc *queue_alloc, Buffer *dstBuffer, uint64_t offset,
                   uint64_t size, uint32_t data);
  Vulten_pipeline *get_cached_pipeline(std::string pipe_string);
  Vulten_pipeline *create_pipeline(
      std::string pipe_string, std::vector<vk::DescriptorType> buffer_types,
      std::vector<uint32_t> shader_spv, vk::SpecializationInfo *spec_info = {},
      std::vector<vk::PushConstantRange> push_ranges = {});
  Queue_alloc get_queue(bool graphics, bool compute, bool transfer,
                        bool sparse);
  Descriptor_set_alloc get_descriptor_sets(int num_descriptors,
                                           vk::DescriptorSetLayout layouts);

  // Instance(const Instance&) = delete;
  Instance(uint32_t dev_num);
  ~Instance();
};

}  // namespace vulten_backend