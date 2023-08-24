#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/reluGrad/ReluGrad_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void ReluGradOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ReluGradOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor gradients = tensor_utills::get_input_tensor(
      "ReluGradOp:gradients", 0, ctx, status.get());

  tensor_utills::Input_tensor features = tensor_utills::get_input_tensor(
      "ReluGradOp:features", 1, ctx, status.get());

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "ReluGradOp:output", 0, gradients.dims, ctx, status.get());

  if (gradients.is_empty || features.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::reluGrad::run_op(inst, (vulten_ops::Data_type)T,
                               gradients.vulten_tensor, features.vulten_tensor,
                               output.vulten_tensor);
}

template <TF_DataType T>
void RegisterReluGradOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("ReluGrad", device_type, nullptr,
                                      &ReluGradOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ReluGrad kernel with attribute T";
  TF_RegisterKernelBuilder("ReluGrad", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering ReluGrad kernel";
}

void RegisterDeviceReluGrad(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterReluGradOpKernel<T>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)
}