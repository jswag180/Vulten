#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/ReluGrad_op.h"
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

  GET_INPUT_TENSOR("ReluGrad", gradients, 0, ctx, status)
  GET_INPUT_TENSOR("ReluGrad", features, 1, ctx, status)

  MAKE_OUTPUT_TENSOR("ReluGrad", output, 0, gradients_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::ReluGrad_op* reluGrad_op = nullptr;
  std::string op_cache_name = "ReluGrad";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::ReluGrad_op(inst);
  }
  reluGrad_op = (vulten_ops::ReluGrad_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  reluGrad_op->run_op((vulten_ops::Data_type)T, gradients_tensor,
                      features_tensor, output_tensor);
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