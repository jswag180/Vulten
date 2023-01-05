#include "gpu/gpu_kernel_init.h"

#include "tensorflow/c/kernels.h"

void RegisterDeviceKernels(const char* device_type) {
  // Ativations
  RegisterDeviceRelu(device_type);

  // Convs

  // Optimizers

  // Misc
}