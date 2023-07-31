#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/pow/Pow_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void PowOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("PowOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor x =
      tensor_utills::get_input_tensor("PowOp:x", 0, ctx, status.get());

  tensor_utills::Input_tensor y =
      tensor_utills::get_input_tensor("PowOp:y", 1, ctx, status.get());

  uint32_t scalar = 0;
  absl::InlinedVector<int64_t, 4>& out_dims = x.dims;
  if (TF_TensorElementCount(x.tf_tensor) <= 1 &&
      !(TF_TensorElementCount(y.tf_tensor) <= 1)) {
    scalar = 1;
    out_dims = y.dims;
  } else if (TF_TensorElementCount(y.tf_tensor) <= 1 &&
             !(TF_TensorElementCount(x.tf_tensor) <= 1)) {
    scalar = 2;
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "PowOp:output", 0, out_dims, ctx, status.get());

  if (x.is_empty || y.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Pow_op* pow_op = nullptr;
  std::string op_cache_name = "Relu";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Pow_op(inst);
  }
  pow_op = (vulten_ops::Pow_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  pow_op->run_op((vulten_ops::Data_type)T, scalar, x.vulten_tensor,
                 y.vulten_tensor, output.vulten_tensor);
}

template <TF_DataType T>
void RegisterPowOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Pow", device_type, nullptr,
                                      &PowOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel with attribute T";
  TF_RegisterKernelBuilder("Pow", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel";
}

void RegisterDevicePow(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterPowOpKernel<T>(device_type);

  CALL_ALL_BASIC_TYPES(REGISTER_KERNEL)
}