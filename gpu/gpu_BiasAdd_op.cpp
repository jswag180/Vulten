#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/BiasAdd_op.h"
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

  GET_INPUT_TENSOR("BiasAdd", input, 0, ctx, status)

  GET_INPUT_TENSOR("BiasAdd", bias, 1, ctx, status)

  MAKE_OUTPUT_TENSOR("BiasAdd", output, 0, input_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::BiasAdd_op* biasAdd_op = nullptr;
  std::string op_cache_name = "BiasAdd";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::BiasAdd_op(inst);
  }
  biasAdd_op = (vulten_ops::BiasAdd_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  biasAdd_op->run_op((vulten_ops::Data_type)T, input_tensor, bias_tensor,
                     uint32_t(bias_dims[0]), output_tensor);
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