#include "gpu/gpu_kernel_init.h"

#include "tensorflow/c/kernels.h"

void RegisterDeviceKernels(const char* device_type) {
  // Ativations
  RegisterDeviceRelu(device_type);
  RegisterDeviceReluGrad(device_type);
  RegisterDeviceSoftmax(device_type);

  // Convs

  // Optimizers
  RegisterResourceApplyAdam(device_type);

  // Misc
  RegisterDeviceAssignVariable(device_type);
  RegisterAssignAddSubVariable(device_type);
  RegisterDeviceBasicOps(device_type);
  RegisterDeviceCast(device_type);
  RegisterDeviceReadVariableOp(device_type);
  RegisterDeviceMatMul(device_type);
  RegisterDeviceBiasAddGradOp(device_type);
  RegisterDeviceReduce(device_type);
  RegisterDeviceAddn(device_type);
  RegisterDeviceIdentity(device_type);
  RegisterDevicePow(device_type);
  RegisterDeviceIdentityN(device_type);
  RegisterDeviceMultiFunc(device_type);
  RegisterDeviceBiasAdd(device_type);
  RegisterDeviceSparseSoftmaxCrossEntropyWithLogits(device_type);
  RegisterDeviceBroadcastOp(device_type);
}