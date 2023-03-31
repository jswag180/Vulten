#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/Vulten_backend_ops.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

void IdentityNOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("IdentityNOp")

  StatusSafePtr status(TF_NewStatus());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  for (int i = 0; i < TF_NumInputs(ctx); i++) {
    TF_Tensor* tensor = nullptr;
    TF_GetInput(ctx, i, &tensor, status.get());
    TensorSafePtr tensor_safe_ptr(tensor);
    if (TF_GetCode(status.get()) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status.get());
      std::cout << "Error: IdentityN at " << i << "\n";
      return;
    }
    absl::InlinedVector<int64_t, 4> tensor_dims(
        TF_NumDims(tensor_safe_ptr.get()));
    for (auto j = 0; j < TF_NumDims(tensor_safe_ptr.get()); ++j) {
      tensor_dims[j] = TF_Dim(tensor_safe_ptr.get(), j);
    }

    uint64_t total_output_tensor_elements = 1;
    for (uint64_t j = 0; j < tensor_dims.size(); j++) {
      total_output_tensor_elements *= tensor_dims[j];
    }
    TensorSafePtr output_safe_ptr(TF_AllocateOutput(
        ctx, i, TF_ExpectedOutputDataType(ctx, i), tensor_dims.data(),
        tensor_dims.size(),
        total_output_tensor_elements * TF_ExpectedOutputDataType(ctx, i),
        status.get()));
    if (TF_GetCode(status.get()) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status.get());
      std::cout << "Error: IdentityN out at " << i << "\n";
      return;
    }

    inst->copy_buffer(
        VOID_TO_DEVICE_BUFFER(TF_TensorData(tensor_safe_ptr.get())),
        VOID_TO_DEVICE_BUFFER(TF_TensorData(output_safe_ptr.get())));
  }
}

void RegisterIdentityNOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("IdentityN", device_type, nullptr,
                                      &IdentityNOp_Compute, nullptr);

  TF_RegisterKernelBuilder("IdentityN", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering IdentityN kernel";
}

void RegisterDeviceIdentityN(const char* device_type) {
  RegisterIdentityNOpKernel(device_type);
}