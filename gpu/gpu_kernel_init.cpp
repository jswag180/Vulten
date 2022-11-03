#include "gpu/gpu_kernel_init.h"

#include "tensorflow/c/kernels.h"

void RegisterDeviceKernels(const char* device_type) {
  // Ativations
  RegisterDeviceRelu(device_type);
  RegisterDeviceSoftmax(device_type);

  // Convs
  RegisterDeviceConv2D(device_type);

  // Optimizers
  RegisterResourceApplyAdam(device_type);

  // Misc
  RegisterDeviceAssignVariable(device_type);
  // RegisterDeviceReadVariableOp(device_type);
  RegisterAssignAddVariable(device_type);
  // RegisterDeviceIdentityOp(device_type);
  RegisterDeviceBasicOps(device_type);
  RegisterDevicePow(device_type);
  RegisterDeviceCast(device_type);
  // RegisterDeviceStridedSliceOp(device_type);
  // RegisterDeviceSparseSoftmaxCrossEntropyWithLogitsOp(device_type);

  RegisterDeviceBiasAdd(device_type);
  RegisterDeviceMatMul(device_type);
}