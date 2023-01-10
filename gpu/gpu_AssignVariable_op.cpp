#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

void AssignVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("AssignVariableOp")

  StatusSafePtr status(TF_NewStatus());

  TF_AssignVariable(ctx, 0, 1, false, &tensor_utills::copyFunc, status.get());
}

void RegisterAssignVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("AssignVariableOp", device_type, nullptr,
                                      &AssignVariableOp_Compte, nullptr);
  TF_RegisterKernelBuilder("AssignVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignVariableOp kernel";
}

void RegisterDeviceAssignVariable(const char* device_type) {
  RegisterAssignVariableOp(device_type);
}