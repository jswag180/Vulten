#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/softmax/Softmax_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void SoftmaxOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("SoftmaxOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("SoftmaxOp:input", 0, ctx, status.get());

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "SoftmaxOp:output", 0, input.dims, ctx, status.get());

  if (input.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Softmax_op* softmax_op = nullptr;
  std::string op_cache_name = "Softmax";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Softmax_op(inst);
  }
  softmax_op = (vulten_ops::Softmax_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  softmax_op->run_op((vulten_ops::Data_type)T, input.vulten_tensor,
                     output.vulten_tensor);
}

template <TF_DataType T>
void RegisterSoftmaxOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Softmax", device_type, nullptr,
                                      &SoftmaxOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Softmax kernel with attribute T";
  TF_RegisterKernelBuilder("Softmax", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Softmax kernel";
}

void RegisterDeviceSoftmax(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterSoftmaxOpKernel<T>(device_type);

  REGISTER_KERNEL(TF_FLOAT)
}