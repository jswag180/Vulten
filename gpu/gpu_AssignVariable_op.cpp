#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "gpu_variable_helpers.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

namespace vulten_plugin {

template <typename T>
void AssignVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  StatusSafePtr status(TF_NewStatus());

  TF_AssignVariable(ctx, 0, 1, false, &varHelpers::copyFunc, status.get());
}

template <typename T>
void RegisterAssignVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("AssignVariableOp", device_type, nullptr,
                                      &AssignVariableOp_Compte<T>, nullptr);
  TF_RegisterKernelBuilder("AssignVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignVariableOp kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceAssignVariable(const char* device_type) {
  vulten_plugin::RegisterAssignVariableOp<float>(device_type);
}