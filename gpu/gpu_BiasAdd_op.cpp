#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/biasAdd/BiasAdd_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void BiasAddOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BiasAddOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("BiasAddOp:input", 0, ctx, status.get());
  if (input.is_empty) {
    tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
        "BiasAddOp:output", 0, input.dims, ctx, status.get());

    return;
  }

  tensor_utills::Input_tensor bias =
      tensor_utills::get_input_tensor("BiasAddOp:bias", 1, ctx, status.get());
  if (bias.is_empty) {
    return;
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "BiasAddOp:output", 0, input.dims, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::bias_add::run_op(inst, (vulten_ops::Data_type)T,
                               input.vulten_tensor, bias.vulten_tensor,
                               uint32_t(bias.dims[0]), output.vulten_tensor);
}

template <TF_DataType T>
void RegisterBiasAddOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("BiasAdd", device_type, nullptr,
                                      &BiasAddOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAdd kernel with attribute T";
  TF_RegisterKernelBuilder("BiasAdd", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAdd kernel";
}

void RegisterDeviceBiasAdd(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterBiasAddOpKernel<T>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)
}