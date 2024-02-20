#include <exception>

#include "Vulten_backend.h"
#include "vulten_logger.h"

namespace vulten_backend {

Descriptor_set_alloc Instance::get_descriptor_sets(
    int num_descriptors, vk::DescriptorSetLayout layout) {
  // Look for available desctriptors
  {
    std::shared_lock lock(descriptor_mutex);
    for (uint32_t i = 0; i < descriptors.size(); i++) {
      if (descriptors[i]->descriptors == num_descriptors) {
        auto hasLock = descriptors[i]->mutex->try_lock();
        if (hasLock) {
          return Descriptor_set_alloc(descriptors[i]);
        }
      }
    }
  }

  // There is either no desctriptor that meets the req or is in use
  std::unique_lock lock(descriptor_mutex);
  std::shared_ptr<Descriptor_set> descriptor_set =
      std::shared_ptr<Descriptor_set>(new Descriptor_set());
  descriptor_set->descriptors = num_descriptors;
  vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1,
                                                          &layout);
  try {
    descriptor_set->vk_descriptor_set =
        logical_dev.allocateDescriptorSets(descriptor_set_alloc_info).front();
  } catch (std::exception e) {
    VULTEN_LOG_ERROR("Failed to allocate descriptor sets!")
    exit(-1);
  }
  descriptor_set->mutex->lock();
  descriptors.push_back(descriptor_set);

  return Descriptor_set_alloc(descriptors.back());
}

}  // namespace vulten_backend