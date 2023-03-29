#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/Exp_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void ExpOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ExpOp")

  StatusSafePtr status(TF_NewStatus());

  GET_INPUT_TENSOR("Exp", input, 0, ctx, status)

  MAKE_OUTPUT_TENSOR("Exp", output, 0, input_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Exp_op* exp_op = nullptr;
  std::string op_cache_name = "Exp";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Exp_op(inst);
  }
  exp_op = (vulten_ops::Exp_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  exp_op->run_op((vulten_ops::Data_type)T, input_tensor, output_tensor);
}

template <TF_DataType T>
void RegisterExpOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Exp", device_type, nullptr,
                                      &ExpOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel with attribute T";
  TF_RegisterKernelBuilder("Exp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel";
}

void RegisterDeviceExp(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterExpOpKernel<T>(device_type);

  // Exp is hard with compute shaders with the lack of native
  // pow, exp, cos, and sin in extended types.
  REGISTER_KERNEL(TF_FLOAT)
}