#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

void ReadVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ReadVariableOp")

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor** ref = new TF_Tensor*;

  TF_GetInputTensorFromVariable(ctx, 0, 1, 0, 0, &tensor_utills::copyFunc, ref,
                                status.get());

  TF_SetOutput(ctx, 0, *ref, status.get());

  delete ref;
}

void RegisterReadVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("ReadVariableOp", device_type, nullptr,
                                      &ReadVariableOp_Compte, nullptr);
  TF_RegisterKernelBuilder("ReadVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ReadVariableOp kernel";
}

void RegisterDeviceReadVariableOp(const char* device_type) {
  RegisterReadVariableOp(device_type);
}