#include "Vulten_backend.h"

#include <vulkan/vulkan_structs.hpp>
#define VMA_IMPLEMENTATION
#include "VulkanMemoryAllocator/include/vk_mem_alloc.h"

namespace instance_utill {
static vk::Instance instance;
static bool is_initialized = false;
static std::mutex instance_mutex;
}  // namespace instance_utill

namespace device_property_utill {
static std::vector<vulten_backend::Device_property> devices;
static bool is_initialized = false;
static std::mutex devices_mutex;
}  // namespace device_property_utill

namespace vulten_backend {

Vk_instance::Vk_instance() {
  instance_utill::instance_mutex.lock();

  if (instance_utill::is_initialized) {
    instance = &instance_utill::instance;
  } else {
    VULTEN_LOG_INFO("Creating vk::Instance")
    vk::ApplicationInfo AppInfo{
        "Vulten",           // Application Name
        1,                  // Application Version
        nullptr,            // Engine Name or nullptr
        0,                  // Engine Version
        VK_API_VERSION_1_2  // Vulkan API version
    };

    const std::vector<const char *> instExtend = {
        "VK_KHR_portability_enumeration"};

#ifdef NDEBUG
    vk::InstanceCreateInfo InstanceCreateInfo(
        vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,  // Flags
        &AppInfo,            // Application Info
        0,                   // Layers count
        nullptr,             // Layers
        instExtend.size(),   // Instance extention count
        instExtend.data());  // Instance extentions
#else
    const std::vector<const char *> Layers = {"VK_LAYER_KHRONOS_validation"};
    vk::InstanceCreateInfo InstanceCreateInfo(
        vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,  // Flags
        &AppInfo,            // Application Info
        Layers.size(),       // Layers count
        Layers.data(),       // Layers
        instExtend.size(),   // Instance extention count
        instExtend.data());  // Instance extentions

#endif
    instance_utill::instance = vk::createInstance(InstanceCreateInfo);
    instance = &instance_utill::instance;
    instance_utill::is_initialized = true;
  }
}

Vk_instance::~Vk_instance() { instance_utill::instance_mutex.unlock(); }

Device_propertys::Device_propertys() {
  device_property_utill::devices_mutex.lock();

  devices = &device_property_utill::devices;

  if (device_property_utill::is_initialized) {
    return;
  } else {
    VULTEN_LOG_INFO("Populating Device_propertys")
    Vk_instance vk_inst = Vk_instance();

    auto physicalDevices = vk_inst.instance->enumeratePhysicalDevices();

    if (physicalDevices.size() <= 0) {
      VULTEN_LOG_ERROR("enumeratePhysicalDevices returned no devices")
    }

    for (uint32_t i = 0; i < physicalDevices.size(); i++) {
      VULTEN_LOG_INFO("found device: "
                      << i << " "
                      << physicalDevices[i].getProperties().deviceName)

      Device_property dev_prop = Device_property();

      dev_prop.props = physicalDevices[i].getProperties();

      auto prop2_chain =
          physicalDevices[i]
              .getProperties2<vk::PhysicalDeviceProperties2,
                              vk::PhysicalDeviceSubgroupProperties>();

      dev_prop.props2 = prop2_chain.get<vk::PhysicalDeviceProperties2>();

      auto sub_group_props =
          prop2_chain.get<vk::PhysicalDeviceSubgroupProperties>();
      dev_prop.subgroupSize = sub_group_props.subgroupSize;

      auto queueFamilyProperties =
          physicalDevices[i].getQueueFamilyProperties();
      std::vector<Device_queue_prop> dev_queue_props =
          std::vector<Device_queue_prop>(queueFamilyProperties.size());
      for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        dev_queue_props[i].max_queues = queueFamilyProperties[i].queueCount;

        dev_queue_props[i].hasGraphics =
            queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics
                ? true
                : false;
        dev_queue_props[i].hasCompute =
            queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute
                ? true
                : false;
        dev_queue_props[i].hasTransfer =
            queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eTransfer
                ? true
                : false;
        dev_queue_props[i].hasSparse = queueFamilyProperties[i].queueFlags &
                                               vk::QueueFlagBits::eSparseBinding
                                           ? true
                                           : false;
      }
      dev_prop.queue_props = dev_queue_props;

      dev_prop.mem_props = physicalDevices[i].getMemoryProperties();

      std::vector<std::string> extens;
      auto exten_props =
          physicalDevices[i].enumerateDeviceExtensionProperties();
      for (uint32_t i = 0; i < exten_props.size(); i++) {
        extens.push_back(std::string(
            static_cast<const char *>(exten_props[i].extensionName)));
      }
      dev_prop.extens = extens;

      devices->push_back(dev_prop);
    }
    device_property_utill::is_initialized = true;
  }
}

Device_propertys::~Device_propertys() {
  device_property_utill::devices_mutex.unlock();
}

bool enable_if_avaliable(const char *exten_name,
                         std::vector<std::string> &extents) {
  if (std::find(extents.begin(), extents.end(), exten_name) != extents.end()) {
    extents.push_back(exten_name);
    return true;
  } else {
    return false;
  }
}

Instance::Instance(uint32_t dev_num) {
  VULTEN_LOG_INFO("Creating vulten_backend::Instance")

  device_num = dev_num;
  {
    Vk_instance inst = Vk_instance();
    physical_dev = inst.instance->enumeratePhysicalDevices()[device_num];
  }

  vulten_backend::Device_propertys dev_props =
      vulten_backend::Device_propertys();
  device_propertys = (*dev_props.devices)[device_num];
  std::vector<vk::DeviceQueueCreateInfo> queues_info =
      std::vector<vk::DeviceQueueCreateInfo>(
          device_propertys.queue_props.size());

  static float prio = 1.0f;
  total_queues = 0;
  for (uint32_t i = 0; i < device_propertys.queue_props.size(); i++) {
    auto queue_info = vk::DeviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), i,
        device_propertys.queue_props[i].max_queues, &prio);
    total_queues += device_propertys.queue_props[i].max_queues;
    queues_info[i] = queue_info;
  }
  queues = new Queue[total_queues];

  std::vector<const char *> extens = {};

  enable_if_avaliable("VK_KHR_portability_subset", device_propertys.extens);
  bool has_mem_budget =
      enable_if_avaliable("VK_EXT_memory_budget", device_propertys.extens);
  bool has_mem_req2 = enable_if_avaliable("VK_KHR_get_memory_requirements2",
                                          device_propertys.extens);
  bool has_ded_alloc = enable_if_avaliable("VK_KHR_dedicated_allocation",
                                           device_propertys.extens);
  bool has_amd_mem = enable_if_avaliable("VK_AMD_device_coherent_memory",
                                         device_propertys.extens);

  vk::PhysicalDeviceShaderFloat16Int8Features half_char_feat =
      vk::PhysicalDeviceShaderFloat16Int8Features();
  half_char_feat.setShaderFloat16(true);
  half_char_feat.setShaderInt8(true);

  vk::PhysicalDevice8BitStorageFeatures char_buffer_feat =
      vk::PhysicalDevice8BitStorageFeatures();
  char_buffer_feat.setStoragePushConstant8(true);
  char_buffer_feat.setStorageBuffer8BitAccess(true);
  char_buffer_feat.setPNext(&half_char_feat);

  vk::PhysicalDevice16BitStorageFeatures half_buffer_feat =
      vk::PhysicalDevice16BitStorageFeatures();
  half_buffer_feat.setStorageBuffer16BitAccess(true);
  half_buffer_feat.setStorageInputOutput16(true);
  half_buffer_feat.setPNext(&char_buffer_feat);

  vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures extend_sub =
      vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures();
  extend_sub.setShaderSubgroupExtendedTypes(true);
  extend_sub.setPNext(&half_buffer_feat);

  vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_block_feat =
      vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT();
  scalar_block_feat.setScalarBlockLayout(true);
  scalar_block_feat.setPNext(&extend_sub);

  vk::PhysicalDeviceCoherentMemoryFeaturesAMD amd_mem_feat =
      vk::PhysicalDeviceCoherentMemoryFeaturesAMD();
  amd_mem_feat.setDeviceCoherentMemory(has_amd_mem);
  amd_mem_feat.setPNext(&scalar_block_feat);

  vk::PhysicalDeviceFeatures2 dev_features2 = vk::PhysicalDeviceFeatures2();
  dev_features2.features.setShaderInt16(true);
  dev_features2.features.setShaderInt64(true);
  dev_features2.features.setShaderFloat64(true);
  dev_features2.setPNext(&amd_mem_feat);

  auto dev_create_info = vk::DeviceCreateInfo(
      vk::DeviceCreateFlags(), queues_info.size(), queues_info.data(), 0, {},
      extens.size(), extens.data(), nullptr, &dev_features2);

  logical_dev = physical_dev.createDevice(dev_create_info);

  pipeline_cache =
      logical_dev.createPipelineCache(vk::PipelineCacheCreateInfo());

  for (int i = 0; i < queues_info.size(); i++) {
    for (int j = 0; j < queues_info[i].queueCount; j++) {
      queues[i + j].vk_queue =
          logical_dev.getQueue(queues_info[i].queueFamilyIndex, j);

      vk::CommandPoolCreateInfo cmd_pool_create_info(
          vk::CommandPoolCreateFlagBits::eTransient |
              vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          queues_info[i].queueFamilyIndex);
      queues[i + j].cmd_pool =
          logical_dev.createCommandPool(cmd_pool_create_info);
      queues[i + j].graphics = device_propertys.queue_props[i].hasGraphics;
      queues[i + j].compute = device_propertys.queue_props[i].hasCompute;
      queues[i + j].transfer = device_propertys.queue_props[i].hasTransfer;
      queues[i + j].sparse = device_propertys.queue_props[i].hasSparse;
    }
  }

  {
    Vk_instance inst = Vk_instance();

    VmaAllocatorCreateInfo vmaAllocatorCreateInfo = VmaAllocatorCreateInfo();
    vmaAllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    vmaAllocatorCreateInfo.physicalDevice = physical_dev;
    vmaAllocatorCreateInfo.device = logical_dev;
    vmaAllocatorCreateInfo.instance = *inst.instance;
    if (has_mem_budget) {
      vmaAllocatorCreateInfo.flags |=
          VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (has_mem_req2 && has_ded_alloc) {
      vmaAllocatorCreateInfo.flags |=
          VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }
    if (has_amd_mem) {
      vmaAllocatorCreateInfo.flags |=
          VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    }

    vmaCreateAllocator(&vmaAllocatorCreateInfo, &allocator);
  }

  vk::DescriptorPoolSize descriptor_pool_size(
      vk::DescriptorType::eStorageBuffer, 32);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
      vk::DescriptorPoolCreateFlags(), 32, descriptor_pool_size);
  descriptor_pool =
      logical_dev.createDescriptorPool(descriptor_pool_create_info);
}

Host_mappable_buffer *Instance::create_host_mappable_buffer(
    void *data, uint32_t size, bool sync_to_device, bool trans_src,
    bool trans_dst, bool staging, bool uniform) {
  return new Host_mappable_buffer(this, data, size, sync_to_device, trans_src,
                                  trans_dst, staging, uniform);
}

Device_buffer *Instance::create_device_buffer(uint32_t size, bool trans_src,
                                              bool trans_dst) {
  return new Device_buffer(this, size, trans_src, trans_dst);
}

void Instance::copy_buffer(Buffer *src, Buffer *dest, uint32_t size) {
  Queue_alloc queue_alloc = get_queue(false, false, true, false);

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buff.begin(cmd_buff_begin_info);

  vk::BufferCopy buff_copy;
  if (size > 0) {
    buff_copy = vk::BufferCopy(0, 0, size);
  } else {
    buff_copy =
        vk::BufferCopy(0, 0, std::min(src->buffer_size, dest->buffer_size));
  }
  cmd_buff.copyBuffer(src->vk_buffer, dest->vk_buffer, buff_copy);

  cmd_buff.end();

  vk::SubmitInfo submit_info({}, {}, {}, 1, &cmd_buff);
  queue_alloc.queue->vk_queue.submit(submit_info, {});
  queue_alloc.queue->vk_queue.waitIdle();
  logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buff);
}

void Instance::copy_buffer(Queue_alloc *queue_alloc, Buffer *src, Buffer *dest,
                           uint32_t size) {
  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc->queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buff.begin(cmd_buff_begin_info);

  vk::BufferCopy buff_copy;
  if (size > 0) {
    buff_copy = vk::BufferCopy(0, 0, size);
  } else {
    buff_copy =
        vk::BufferCopy(0, 0, std::min(src->buffer_size, dest->buffer_size));
  }
  cmd_buff.copyBuffer(src->vk_buffer, dest->vk_buffer, buff_copy);

  cmd_buff.end();

  vk::SubmitInfo submit_info({}, {}, {}, 1, &cmd_buff);
  queue_alloc->queue->vk_queue.submit(submit_info, {});
  queue_alloc->queue->vk_queue.waitIdle();
  logical_dev.freeCommandBuffers(queue_alloc->queue->cmd_pool, cmd_buff);
}

void Instance::fill_buffer(Buffer *dstBuffer, uint64_t offset, uint64_t size,
                           uint32_t data) {
  Queue_alloc queue_alloc = get_queue(false, false, true, false);

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc.queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buff.begin(cmd_buff_begin_info);

  cmd_buff.fillBuffer(dstBuffer->vk_buffer, offset, size, data);

  cmd_buff.end();

  vk::SubmitInfo submit_info({}, {}, {}, 1, &cmd_buff);
  queue_alloc.queue->vk_queue.submit(submit_info, {});
  queue_alloc.queue->vk_queue.waitIdle();
  logical_dev.freeCommandBuffers(queue_alloc.queue->cmd_pool, cmd_buff);
}

void Instance::fill_buffer(Queue_alloc *queue_alloc, Buffer *dstBuffer,
                           uint64_t offset, uint64_t size, uint32_t data) {
  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      queue_alloc->queue->cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buff.begin(cmd_buff_begin_info);

  cmd_buff.fillBuffer(dstBuffer->vk_buffer, offset, size, data);

  cmd_buff.end();

  vk::SubmitInfo submit_info({}, {}, {}, 1, &cmd_buff);
  queue_alloc->queue->vk_queue.submit(submit_info, {});
  queue_alloc->queue->vk_queue.waitIdle();
  logical_dev.freeCommandBuffers(queue_alloc->queue->cmd_pool, cmd_buff);
}

Instance::~Instance() {
  for (int i = 0; i < total_queues; i++) {
    logical_dev.destroyCommandPool(queues[i].cmd_pool);
  }
  delete[] queues;
  vmaDestroyAllocator(allocator);
  logical_dev.destroyPipelineCache(pipeline_cache);
}

Vulten_pipeline *Instance::get_cached_pipeline(std::string pipe_string) {
  std::shared_lock lock(pipe_mutex);
  auto pipeline = pipelines.find(pipe_string);
  if (pipeline == pipelines.end()) return nullptr;
  return (*pipeline).second;
}

Queue_alloc Instance::get_queue(bool graphics, bool compute, bool transfer,
                                bool sparse) {
  while (true) {
    for (int i = 0; i < total_queues; i++) {
      if (graphics == true) {
        if (queues[i].graphics == false) {
          continue;
        }
      }
      if (compute == true) {
        if (queues[i].compute == false) {
          continue;
        }
      }
      if (transfer == true) {
        if (queues[i].transfer == false) {
          continue;
        }
      }
      if (sparse == true) {
        if (queues[i].sparse == false) {
          continue;
        }
      }
      auto hasLock = queues[i].queue_mutex.try_lock();
      if (hasLock) {
        return Queue_alloc(&queues[i]);
      }
    }
  }
}

Vulten_pipeline *Instance::create_pipeline(
    std::string pipe_string, std::vector<vk::DescriptorType> buffer_types,
    std::vector<uint32_t> shader_spv, vk::SpecializationInfo *spec_info,
    std::vector<vk::PushConstantRange> push_ranges) {
  std::unique_lock lock(pipe_mutex);

  pipelines[pipe_string] = new Vulten_pipeline(this, buffer_types, shader_spv,
                                               spec_info, push_ranges);
  return pipelines[pipe_string];
}

}  // namespace vulten_backend