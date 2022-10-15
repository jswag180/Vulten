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

// struct StatusDeleter {
//     void operator()(TF_Status *s) {
//         if (s != nullptr) {
//             TF_DeleteStatus(s);
//         }
//     }
// };

// struct TensorDeleter {
//     void operator()(TF_Tensor *t) {
//         if (t != nullptr) {
//             TF_DeleteTensor(t);
//         }
//     }
// };

// using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
// using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

// https://www.tensorflow.org/api_docs/python/tf/raw_ops/AssignVariableOp
namespace vulten_plugin {

template <typename T>
void AssignVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  StatusSafePtr status(TF_NewStatus());

  // TF_GetInputTensorFromVariable()
  TF_AssignVariable(ctx, 0, 1, false, &varHelpers::copyFunc, status.get());
  // TF_AssignUpdateVariable()
  ////TF_AssignRefVariable()
}

template <typename T>
void RegisterAssignVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("AssignVariableOp", device_type, nullptr,
                                      &AssignVariableOp_Compte<T>, nullptr);
  // TF_KernelBuilder_TypeConstraint(builder, "T", TF_VARIANT, status.get());
  // if (TF_OK != TF_GetCode(status.get()))
  //     std::cout << " Error while registering AssignVariableOp kernel with
  //     attribute T";
  TF_RegisterKernelBuilder("AssignVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignVariableOp kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceAssignVariable(const char* device_type) {
  vulten_plugin::RegisterAssignVariableOp<float>(device_type);
}