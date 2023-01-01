#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#if VULTEN_LOG_LEVEL == 0
#define VULTEN_LOG_INFO(M)
#define VULTEN_LOG_DEBUG(M)
#elif VULTEN_LOG_LEVEL == 1
#define VULTEN_LOG_INFO(M) std::cout << "Vulten [INFO]: " << M << "\n";
#define VULTEN_LOG_DEBUG(M)
#elif VULTEN_LOG_LEVEL == 2
#define VULTEN_LOG_INFO(M) std::cout << "Vulten [INFO]: " << M << "\n";
#define VULTEN_LOG_DEBUG(M) std::cout << "Vulten [DEBUG]: " << M << "\n";
#endif
#define VULTEN_LOG_ERROR(M) std::cout << "Vulten [ERROR]: " << M << "\n";

#define VOID_TO_INSTANCE(X) static_cast<vulten_backend::Instance *>(X)
#define VOID_TO_DEVICE_BUFFER(X) static_cast<vulten_backend::Device_buffer *>(X)
#define VOID_TO_HOST_MAPPABLE_BUFFER(X) \
  static_cast<vulten_backend::Host_mappable_buffer *>(X)

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

struct Mapped_memory {
 private:
  vk::Device *m_device;
  vk::DeviceMemory *m_memory;

 public:
  uint8_t *data;
  uint32_t size;

  Mapped_memory(vk::Device &device, vk::DeviceMemory &memory,
                uint32_t buffer_size);
  ~Mapped_memory();
};

class Instance;
struct Buffer {
 private:
 public:
  Instance *inst;
  vk::Buffer vk_buffer;
  vk::MemoryRequirements memory_req;
  vk::DeviceMemory device_memory;
  uint32_t buffer_size;

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties,
                          vk::PhysicalDeviceMemoryProperties &memProperties);

  Buffer(Instance *inst, uint32_t size);
  ~Buffer();
};

struct Host_mappable_buffer : Buffer {
 private:
  //
 public:
  Mapped_memory map_to_host();

  Host_mappable_buffer(Instance *instance, uint8_t *data, uint32_t size,
                       bool sync_to_device);
  ~Host_mappable_buffer();
};

struct Device_buffer : Buffer {
 private:
  //
 public:
  Device_buffer(Instance *instance, uint32_t size);
  ~Device_buffer();
};

class Instance {
 private:
  //
 public:
  uint32_t device_num;
  vk::PhysicalDevice physical_dev;
  vk::Device logical_dev;
  std::mutex main_queue_mutex;
  vk::Queue main_queue;
  vk::CommandPool cmd_pool;

  Host_mappable_buffer *create_host_mappable_buffer(uint8_t *data,
                                                    uint32_t size,
                                                    bool sync_to_device = true);
  Device_buffer *create_device_buffer(uint32_t size);
  void copy_buffer(Buffer *src, Buffer *dest);

  // Instance(const Instance&) = delete;
  Instance(uint32_t dev_num);
  ~Instance();
};

}  // namespace vulten_backend