#include "gpuBackend.h"

#include <iostream>

int gpuBackend::numDevices = -1;

std::vector<gpuBackend*> gpuBackend::instances;

std::vector<devicePropertys> gpuBackend::deviceProps;

void gpuBackend::vultenLog(const int level, const char* mseg) {
  if (level == INFO) {
    std::cout << "Vulten [INFO]: " << mseg << "\n";
  } else if (level == ERROR) {
    std::cerr << "Vulten [ERROR]: " << mseg << "\n";
  }
}

int gpuBackend::listDevices() {
  if (numDevices == -1) {
    {
      kp::Manager mgr;
      std::vector<vk::PhysicalDevice> devices = mgr.listDevices();

      for (vk::PhysicalDevice device : devices) {
        devicePropertys newProp;

        newProp.physicalProperties = device.getProperties();

        for (auto exten : device.enumerateDeviceExtensionProperties()) {
          newProp.deviceExtentions.push_back(exten.extensionName);
        }

        newProp.memProperties = device.getMemoryProperties();

        std::vector<QueueProps> queues;
        auto queueProps = device.getQueueFamilyProperties();

        for (int i = 0; i < queueProps.size(); i++) {
          QueueProps prop = QueueProps();
          prop.family = i;
          prop.queues = queueProps[i].queueCount;
          if (queueProps[i].queueFlags & vk::QueueFlagBits::eCompute) {
            prop.hasCompute = true;
          }
          if (queueProps[i].queueFlags & vk::QueueFlagBits::eTransfer) {
            prop.hasTransfer = true;
          }
          if (queueProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            prop.hasGraphics = true;
          }

          queues.push_back(prop);
        }
        newProp.queueProperties = queues;

        deviceProps.push_back(newProp);
      }

      numDevices = devices.size();
    }
  }

  return numDevices;
}

kp::Manager* gpuBackend::getManager() { return mngr; }

std::shared_ptr<kp::TensorT<float>>* gpuBackend::addBuffer(uint64_t size) {
  std::unique_lock lock(buffers_mutex);

  // size / 4 to go from bytes to floats of 4 bytes
  std::shared_ptr<kp::TensorT<float>> ten =
      mngr->tensorT<float>(std::vector<float>(std::ceil(size / 4.0F), 0),
                           kp::Tensor::TensorTypes::eDevice);

  tensors.insert(std::make_pair(ten.get(), ten));

  return &tensors[ten.get()];
}

std::shared_ptr<kp::TensorT<float>>* gpuBackend::getBuffer(void* tensorPtr) {
  return nullptr;
}

bool gpuBackend::isDeviceBuffer(void* tensorPtr) {
  std::unique_lock lock(buffers_mutex);

  if (tensors.count(tensorPtr) ==
      0) {
    return false;
  } else {
    return true;
  }
}

void gpuBackend::removeBuffer(void* tensorPtr) {
  std::unique_lock lock(buffers_mutex);
  static_cast<kp::TensorT<float>*>(tensorPtr)->destroy();
  tensors.erase(tensorPtr);
}

gpuBackend::gpuBackend(int deviceNum) {
  device = deviceNum;

  std::vector<uint32_t> familyIndices;
  for (int i = 0; i < deviceProps[device].queueProperties.size(); i++) {
    if (deviceProps[device].queueProperties[i].isFullyEnabled() &&
        mainQueue == -1) {
      familyIndices.push_back(i);
      mainQueue = familyIndices.size() - 1;
    } else if (deviceProps[device].queueProperties[i].hasTransfer &&
               memoryQueue == -1) {
      familyIndices.push_back(i);
      memoryQueue = familyIndices.size() - 1;
      hasMemQueue = true;
    } else if (deviceProps[device].queueProperties[i].hasTransfer &&
               transferQueue == -1) {
      familyIndices.push_back(i);
      transferQueue = familyIndices.size() - 1;
      hasTransQueue = true;
    }
  }

  mngr = new kp::Manager(device, familyIndices,
                         deviceProps[device].deviceExtentions);

  instances.push_back(this);
}

gpuBackend::~gpuBackend() {
  std::cout << "DESTROY\n";

  delete mngr;

  for (int i = 0; i < instances.size(); i++) {
    if (instances[i] == this) {
      instances.erase(instances.begin() + i);
      break;
    }
  }
}