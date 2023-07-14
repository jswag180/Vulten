#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

#ifndef VULTEN_DISABLE_16BIT
#define CALL_HALF(func) func(TF_HALF)
#define CALL_INT16(func) func(TF_INT16)
#define CALL_UINT16(func) func(TF_UINT16)
#else
#define CALL_HALF(func)
#define CALL_INT16(func)
#define CALL_UINT16(func)
#endif

#ifndef VULTEN_DISABLE_8BIT
#define CALL_INT8(func) func(TF_INT8)
#define CALL_UINT8(func) func(TF_UINT8)
#define CALL_BOOL(func) func(TF_BOOL)
#else
#define CALL_INT8(func)
#define CALL_UINT8(func)
#define CALL_BOOL(func)
#endif

#ifndef VULTEN_DISABLE_INT64
#define CALL_INT64(func) func(TF_INT64)
#define CALL_UINT64(func) func(TF_UINT64)
#else
#define CALL_INT8(func)
#define CALL_UINT8(func)
#endif

#ifndef VULTEN_DISABLE_DOUBLE
#define CALL_DOUBLE(func) func(TF_DOUBLE)
#define CALL_COMPLEX128(func) func(TF_COMPLEX128)
#else
#define CALL_DOUBLE(func)
#define CALL_COMPLEX128(func)
#endif

#define CALL_ALL_BASIC_TYPES(func)                                      \
  func(TF_FLOAT) CALL_HALF(func) CALL_DOUBLE(func) func(TF_INT32)       \
      func(TF_UINT32) CALL_INT8(func) CALL_UINT8(func) CALL_INT64(func) \
          CALL_INT64(func) CALL_INT16(func) CALL_INT16(func)
#define CALL_COMPLEX(func) func(TF_COMPLEX64) CALL_COMPLEX128(func)
#define CALL_ALL_TYPES(func) CALL_ALL_BASIC_TYPES(func) CALL_COMPLEX(func)
namespace vulten_ops {
class Vulten_op;
};

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

  Buffer(Instance *inst, uint32_t size, bool trans_src, bool trans_dst);
  ~Buffer();
};

struct Host_mappable_buffer : Buffer {
 private:
  //
 public:
  Mapped_memory map_to_host();

  Host_mappable_buffer(Instance *instance, uint8_t *data, uint32_t size,
                       bool sync_to_device, bool trans_src, bool trans_dst);
  ~Host_mappable_buffer();
};

struct Device_buffer : Buffer {
 private:
  //
 public:
  Device_buffer(Instance *instance, uint32_t size, bool trans_src,
                bool trans_dst);
  ~Device_buffer();
};

class Instance {
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
  // opName_Data_type
  std::unordered_map<std::string, vulten_ops::Vulten_op *> op_chache;

  Host_mappable_buffer *create_host_mappable_buffer(uint8_t *data,
                                                    uint32_t size,
                                                    bool sync_to_device = true,
                                                    bool trans_src = true,
                                                    bool trans_dst = true);
  Device_buffer *create_device_buffer(uint32_t size, bool trans_src = true,
                                      bool trans_dst = true);
  void copy_buffer(Buffer *src, Buffer *dest, bool lock = true);
  void fill_buffer(Buffer *dstBuffer, uint64_t offset, uint64_t size,
                   uint32_t data, bool lock = true);

  // Instance(const Instance&) = delete;
  Instance(uint32_t dev_num);
  ~Instance();
};

}  // namespace vulten_backend