#include "Vulten_backend.h"

namespace vulten_backend {
Mapped_memory::Mapped_memory(vk::Device &device, vk::DeviceMemory &memory,
                             uint32_t buffer_size) {
  size = buffer_size;
  m_device = &device;
  m_memory = &memory;
  data = static_cast<uint8_t *>(m_device->mapMemory(*m_memory, 0, size));
}

Mapped_memory::~Mapped_memory() { m_device->unmapMemory(*m_memory); }

// I don't know if I need to lock the main queue mutex for buffer creation.
Buffer::Buffer(Instance *instance, uint32_t size, bool trans_src,
               bool trans_dst) {
  inst = instance;
  buffer_size = size;

  vk::BufferUsageFlags buffer_flags = vk::BufferUsageFlagBits::eStorageBuffer;
  if (trans_src) buffer_flags |= vk::BufferUsageFlagBits::eTransferSrc;
  if (trans_dst) buffer_flags |= vk::BufferUsageFlagBits::eTransferDst;
  vk::BufferCreateInfo buffer_create_info(vk::BufferCreateFlags(), buffer_size,
                                          buffer_flags,
                                          vk::SharingMode::eExclusive, 0);
  vk_buffer = inst->logical_dev.createBuffer(buffer_create_info);

  memory_req = inst->logical_dev.getBufferMemoryRequirements(vk_buffer);
}

uint32_t Buffer::findMemoryType(
    uint32_t typeFilter, vk::MemoryPropertyFlags properties,
    vk::PhysicalDeviceMemoryProperties &memProperties) {
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to allocate buffer memory!");
}

Buffer::~Buffer() {
  inst->logical_dev.destroyBuffer(vk_buffer);
  inst->logical_dev.freeMemory(device_memory);
}

Host_mappable_buffer::Host_mappable_buffer(Instance *instance, uint8_t *data,
                                           uint32_t size, bool sync_to_device,
                                           bool trans_src, bool trans_dst)
    : Buffer(instance, size, trans_src, trans_dst) {
  VULTEN_LOG_DEBUG("Creating vulten_backend::Host_mappable_buffer of size " +
                       std::to_string(buffer_size) + " at "
                   << this)

  vk::PhysicalDeviceMemoryProperties mem_props =
      inst->device_propertys.mem_props;

  vk::MemoryAllocateInfo memory_alloc_info(
      memory_req.size,
      findMemoryType(memory_req.memoryTypeBits,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent,
                     mem_props));

  device_memory = inst->logical_dev.allocateMemory(memory_alloc_info);

  inst->logical_dev.bindBufferMemory(vk_buffer, device_memory, 0);

  if (sync_to_device) {
    Mapped_memory maped_mem =
        Mapped_memory(inst->logical_dev, device_memory, buffer_size);
    memcpy(maped_mem.data, data, buffer_size);
  }
}

Mapped_memory Host_mappable_buffer::map_to_host() {
  return Mapped_memory(inst->logical_dev, device_memory, buffer_size);
}

Host_mappable_buffer::~Host_mappable_buffer(){
    VULTEN_LOG_DEBUG("Freeing vulten_backend::Host_mappable_buffer of size " +
                         std::to_string(buffer_size) + " at "
                     << this)}

Device_buffer::Device_buffer(Instance *instance, uint32_t size, bool trans_src,
                             bool trans_dst)
    : Buffer(instance, size, trans_src, trans_dst) {
  VULTEN_LOG_DEBUG("Creating vulten_backend::Device_buffer of size " +
                       std::to_string(buffer_size) + " at "
                   << this)

  vk::PhysicalDeviceMemoryProperties mem_props =
      (*Device_propertys().devices)[inst->device_num].mem_props;

  vk::MemoryAllocateInfo memory_alloc_info(
      memory_req.size,
      findMemoryType(memory_req.memoryTypeBits,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, mem_props));

  device_memory = inst->logical_dev.allocateMemory(memory_alloc_info);

  inst->logical_dev.bindBufferMemory(vk_buffer, device_memory, 0);
}

Device_buffer::~Device_buffer() {
  VULTEN_LOG_DEBUG("Freeing vulten_backend::Device_buffer of size " +
                       std::to_string(buffer_size) + " at "
                   << this)
}
}  // namespace vulten_backend