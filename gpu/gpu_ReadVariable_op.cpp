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
void ReadVariableOp_Compte(void* kernel, TF_OpKernelContext* ctx) {
  // utills::ScopeTimer timer("ReadVariableOp");
  StatusSafePtr status(TF_NewStatus());

  TF_Tensor** ref = new TF_Tensor*;

  TF_GetInputTensorFromVariable(ctx, 0, 1, 0, 0, &varHelpers::copyFunc, ref,
                                status.get());
  // TF_AssignVariable(ctx, 0, 1, false, &varHelpers::copyFunc, status.get());
  // TF_AssignUpdateVariable()
  ////TF_AssignRefVariable()

  if (TF_TensorElementCount(*ref) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(*ref));
  for (auto i = 0; i < TF_NumDims(*ref); ++i) {
    dims[i] = TF_Dim(*ref, i);
  }

  TF_SetOutput(ctx, 0, *ref, status.get());

  // TensorSafePtr output_safe_ptr(TF_AllocateOutput(
  //     ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
  //     TF_TensorElementCount(*ref) * sizeof(T), status.get()));
  // if (TF_GetCode(status.get()) != TF_OK) {
  //     TF_OpKernelContext_Failure(ctx, status.get());
  //     std::cout << "Error: e\n";
  //     return;
  // }

  // auto source_ptr  =
  // static_cast<std::shared_ptr<kp::TensorT<float>>*>(TF_TensorData(*ref));
  // auto dest_ptr    =
  // static_cast<std::shared_ptr<kp::TensorT<float>>*>(TF_TensorData(output_safe_ptr.get()));

  // SP_Stream stream = TF_GetStream(ctx, status.get());
  // //std::lock_guard<std::mutex> guard(stream->instance->testMutex);

  // stream->instance->mngr->sequence()->record<kp::OpTensorCopy>({*source_ptr,
  // *dest_ptr})->eval();

  delete ref;
}

template <typename T>
void RegisterReadVariableOp(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("ReadVariableOp", device_type, nullptr,
                                      &ReadVariableOp_Compte<T>, nullptr);
  // TF_KernelBuilder_TypeConstraint(builder, "T", TF_VARIANT, status.get());
  // if (TF_OK != TF_GetCode(status.get()))
  //     std::cout << " Error while registering ReadVariableOp kernel with
  //     attribute T";
  TF_RegisterKernelBuilder("ReadVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ReadVariableOp kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceReadVariableOp(const char* device_type) {
  vulten_plugin::RegisterReadVariableOp<float>(device_type);
}