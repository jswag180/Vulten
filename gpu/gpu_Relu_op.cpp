#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/Relu_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "shaders/headers/Relu/Relu.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void ReluOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("ReluOp")

  StatusSafePtr status(TF_NewStatus());

  GET_INPUT_TENSOR("Relu", input, 0, ctx, status)

  MAKE_OUTPUT_TENSOR("Relu", output, 0, input_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Relu_op* relu_op = nullptr;
  std::string op_cache_name = "Relu_" + std::to_string(T);
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Relu_op(
            inst, (vulten_ops::Data_type)T);
  }
  relu_op = (vulten_ops::Relu_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  vulten_ops::Vulten_tensor input_tensor(
      VOID_TO_DEVICE_BUFFER(TF_TensorData(input_safe_ptr.get())),
      input_dims.size(), input_dims.data());
  vulten_ops::Vulten_tensor output_tensor(
      VOID_TO_DEVICE_BUFFER(TF_TensorData(output_safe_ptr.get())),
      input_dims.size(), input_dims.data());

  relu_op->run_op(input_tensor, output_tensor);
}

template <TF_DataType T>
void RegisterReluOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Relu", device_type, nullptr,
                                      &ReluOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel with attribute T";
  TF_RegisterKernelBuilder("ReluOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel";
}

void RegisterDeviceRelu(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterReluOpKernel<T>(device_type);

#ifdef RELU_FLOAT
  REGISTER_KERNEL(TF_FLOAT)
#endif
// #ifdef RELU_INT
//   REGISTER_KERNEL(TF_INT32)
// #endif
// #ifdef RELU_UINT
//   REGISTER_KERNEL(TF_UINT32)
// #endif
// #ifdef RELU_INT64_T
//   REGISTER_KERNEL(TF_INT64)
// #endif
// #ifdef RELU_UINT64_T
//   REGISTER_KERNEL(TF_UINT64)
// #endif
// #ifdef RELU_INT8_T
//   REGISTER_KERNEL(TF_INT8)
// #endif
// #ifdef RELU_UINT8_T
//   REGISTER_KERNEL(TF_UINT8)
// #endif
// #ifdef RELU_DOUBLE
//   REGISTER_KERNEL(TF_DOUBLE)
// #endif
}