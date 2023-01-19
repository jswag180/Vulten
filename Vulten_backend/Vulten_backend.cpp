#include "Vulten_backend.h"

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
        VK_API_VERSION_1_3  // Vulkan API version
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

Instance::Instance(uint32_t dev_num) {
  VULTEN_LOG_INFO("Creating vulten_backend::Instance")

  device_num = dev_num;
  {
    Vk_instance inst = Vk_instance();
    physical_dev = inst.instance->enumeratePhysicalDevices()[device_num];
  }

  std::vector<vk::DeviceQueueCreateInfo> queues_info;

  vulten_backend::Device_propertys dev_props =
      vulten_backend::Device_propertys();

  float prio = 1.0f;
  for (uint32_t i = 0; i < (*dev_props.devices)[device_num].queue_props.size();
       i++) {
    if ((*dev_props.devices)[device_num].queue_props[i].hasCompute &&
        (*dev_props.devices)[device_num].queue_props[i].hasTransfer) {
      auto queue_info =
          vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), i, 1);
      queue_info.setQueuePriorities(prio);
      queues_info.push_back(queue_info);
      break;
    }
  }

  std::vector<const char *> extens = {};

  if (std::find((*dev_props.devices)[device_num].extens.begin(),
                (*dev_props.devices)[device_num].extens.end(),
                "VK_KHR_portability_subset") !=
      (*dev_props.devices)[device_num].extens.end()) {
    extens.push_back("VK_KHR_portability_subset");
  }

  auto dev_create_info = vk::DeviceCreateInfo(
      vk::DeviceCreateFlags(), queues_info.size(), queues_info.data(), 0, {},
      extens.size(), extens.data());

  logical_dev = physical_dev.createDevice(dev_create_info);

  main_queue = logical_dev.getQueue(queues_info[0].queueFamilyIndex, 0);

  vk::CommandPoolCreateInfo cmd_pool_create_info(
      vk::CommandPoolCreateFlagBits::eTransient |
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      0);
  cmd_pool = logical_dev.createCommandPool(cmd_pool_create_info);
}

Host_mappable_buffer *Instance::create_host_mappable_buffer(uint8_t *data,
                                                            uint32_t size,
                                                            bool sync_to_device,
                                                            bool trans_src,
                                                            bool trans_dst) {
  return new Host_mappable_buffer(this, data, size, sync_to_device, trans_src,
                                  trans_dst);
}

Device_buffer *Instance::create_device_buffer(uint32_t size, bool trans_src,
                                              bool trans_dst) {
  return new Device_buffer(this, size, trans_src, trans_dst);
}

void Instance::copy_buffer(Buffer *src, Buffer *dest) {
  main_queue_mutex.lock();

  vk::CommandBufferAllocateInfo cmd_buff_alloc_info(
      cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
  vk::CommandBuffer cmd_buff =
      logical_dev.allocateCommandBuffers(cmd_buff_alloc_info)[0];

  vk::CommandBufferBeginInfo cmd_buff_begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buff.begin(cmd_buff_begin_info);

  vk::BufferCopy buff_copy(0, 0, src->buffer_size);
  cmd_buff.copyBuffer(src->vk_buffer, dest->vk_buffer, buff_copy);

  cmd_buff.end();

  vk::SubmitInfo submit_info({}, {}, {}, 1, &cmd_buff);
  main_queue.submit(submit_info, {});
  main_queue.waitIdle();
  logical_dev.freeCommandBuffers(cmd_pool, cmd_buff);
  main_queue_mutex.unlock();
}

Instance::~Instance() { logical_dev.destroyCommandPool(cmd_pool); }

}  // namespace vulten_backend