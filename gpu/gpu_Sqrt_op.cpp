#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/Sqrt_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void SqrtOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("SqrtOp")

  StatusSafePtr status(TF_NewStatus());

  GET_INPUT_TENSOR("Sqrt", input, 0, ctx, status)

  MAKE_OUTPUT_TENSOR("Sqrt", output, 0, input_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Sqrt_op* sqrt_op = nullptr;
  std::string op_cache_name = "Sqrt";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Sqrt_op(inst);
  }
  sqrt_op = (vulten_ops::Sqrt_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  sqrt_op->run_op((vulten_ops::Data_type)T, input_tensor, output_tensor);
}

template <TF_DataType T>
void RegisterSqrtOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Sqrt", device_type, nullptr,
                                      &SqrtOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sqrt kernel with attribute T";
  TF_RegisterKernelBuilder("Sqrt", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sqrt kernel";
}

void RegisterDeviceSqrt(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterSqrtOpKernel<T>(device_type);

  // Sqrt is hard with compute shaders with the lack of native
  // pow, exp, cos, and sin in extended types.
  REGISTER_KERNEL(TF_FLOAT)
}