#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
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
void ReadVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  // utills::ScopeTimer timer("ReadVariableOp");
  StatusSafePtr status(TF_NewStatus());

  TF_Tensor** ref = new TF_Tensor*;

  TF_GetInputTensorFromVariable(ctx, 0, 1, 0, 0, &varHelpers::copyFunc, ref,
                                status.get());

  if (TF_TensorElementCount(*ref) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(*ref));
  for (auto i = 0; i < TF_NumDims(*ref); ++i) {
    dims[i] = TF_Dim(*ref, i);
  }

  TF_SetOutput(ctx, 0, *ref, status.get());

  delete ref;
}

template <typename T>
void RegisterReadVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("ReadVariableOp", device_type, nullptr,
                                      &ReadVariableOp_Compte<T>, nullptr);
  TF_RegisterKernelBuilder("ReadVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ReadVariableOp kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceReadVariableOp(const char* device_type) {
  vulten_plugin::RegisterReadVariableOp<float>(device_type);
}