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

void IdentityOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("IdentityOp")

  StatusSafePtr status(TF_NewStatus());

  GET_INPUT_TENSOR("Identity", input, 0, ctx, status)

  MAKE_OUTPUT_TENSOR("Identity", output, 0, input_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  inst->copy_buffer(input_tensor.buffer, output_tensor.buffer);
}

void RegisterIdentityOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Identity", device_type, nullptr,
                                      &IdentityOp_Compute, nullptr);

  TF_RegisterKernelBuilder("Identity", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Identity kernel";
}

void RegisterDeviceIdentity(const char* device_type) {
  RegisterIdentityOpKernel(device_type);
}