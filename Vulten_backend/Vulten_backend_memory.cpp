#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "Vulten_backend.h"

namespace vulten_backend {

Buffer::Buffer(Instance *instance, uint32_t size, bool trans_src,
               bool trans_dst) {
  inst = instance;
  buffer_size = size;
}

Buffer::~Buffer() {
  VkBuffer buff = static_cast<VkBuffer>(vk_buffer);
  vmaDestroyBuffer(inst->allocator, buff, allocation);
}

Host_mappable_buffer::Host_mappable_buffer(Instance *instance, void *data,
                                           uint32_t size, bool sync_to_device,
                                           bool trans_src, bool trans_dst,
                                           bool staging)
    : Buffer(instance, size, trans_src, trans_dst) {
  VULTEN_LOG_DEBUG("Creating vulten_backend::Host_mappable_buffer of size " +
                       std::to_string(buffer_size) + " at "
                   << this)

  VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (trans_src) {
    bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (trans_dst) {
    bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
  if (staging) {
    allocCreateInfo.flags |=
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  } else {
    allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  }

  VkBuffer buff;
  vmaCreateBuffer(inst->allocator, &bufferInfo, &allocCreateInfo, &buff,
                  &allocation, &allocInfo);
  vk_buffer = buff;

  if (sync_to_device) {
    memcpy(allocInfo.pMappedData, data, buffer_size);
  }
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

  VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (trans_src) {
    bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (trans_dst) {
    bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

  VkBuffer buff;
  vmaCreateBuffer(inst->allocator, &bufferInfo, &allocCreateInfo, &buff,
                  &allocation, &allocInfo);
  vk_buffer = buff;
}

Device_buffer::~Device_buffer() {
  VULTEN_LOG_DEBUG("Freeing vulten_backend::Device_buffer of size " +
                       std::to_string(buffer_size) + " at "
                   << this)
}
}  // namespace vulten_backend