#include <iostream>

#include "Vulten_backend/ops/Bias_add_grad.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

struct BiasAddGradOp {
  BiasAddGradOp() : format("AAAA") {}
  std::string format;
};

void* BiasAddGradOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new BiasAddGradOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrString(ctx, "data_format",
                                        kernel->format.data(), 4, status.get());

  return kernel;
}

void BiasAddGradOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<BiasAddGradOp*>(kernel);
  }
}

template <TF_DataType T>
void BiasAddGradOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BiasAddGradOp")

  StatusSafePtr status(TF_NewStatus());

  BiasAddGradOp* biasAddGradOp_info = static_cast<BiasAddGradOp*>(kernel);

  TF_Tensor* out_backprop = nullptr;
  TF_GetInput(ctx, 0, &out_backprop, status.get());
  TensorSafePtr out_backprop_safe_ptr(out_backprop);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: BiasAddGradOp out_backprop input\n";
    return;
  }
  if (TF_TensorElementCount(out_backprop_safe_ptr.get()) == 0) return;
  auto out_backprop_ptr =
      VOID_TO_DEVICE_BUFFER(TF_TensorData(out_backprop_safe_ptr.get()));

  absl::InlinedVector<int64_t, 4> out_backprop_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(out_backprop_safe_ptr.get()); ++i) {
    out_backprop_dims[i + (4 - TF_NumDims(out_backprop_safe_ptr.get()))] =
        TF_Dim(out_backprop_safe_ptr.get(), i);
  }

  vulten_ops::Vulten_tensor out_backprop_tensor(
      out_backprop_ptr, out_backprop_dims.size(), out_backprop_dims.data());

  std::vector<int64_t> out_dims = {biasAddGradOp_info->format == "NHWC"
                                       ? out_backprop_dims[3]
                                       : out_backprop_dims[1]};
  MAKE_OUTPUT_TENSOR("BiasAddGradOp", output, 0, out_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Bias_add_grad_op* bias_add_grad_op = nullptr;
  std::string op_cache_name = "BiasAddGradOp";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Bias_add_grad_op(inst);
  }
  bias_add_grad_op =
      (vulten_ops::Bias_add_grad_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  bias_add_grad_op->run_op((vulten_ops::Data_type)T, out_backprop_tensor,
                           biasAddGradOp_info->format == "NHWC"
                               ? vulten_ops::Channel_format::NHWC
                               : vulten_ops::Channel_format::NCHW,
                           output_tensor);
}

template <TF_DataType T>
void RegisterBiasAddGradKernels(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("BiasAddGrad", device_type, BiasAddGradOp_Create,
                          &BiasAddGradOp_Compute<T>, &BiasAddGradOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAddGrad kernel with attribute T";
  TF_RegisterKernelBuilder("BiasAddGrad", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAddGrad kernel";
}

void RegisterDeviceBiasAddGradOp(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterBiasAddGradKernels<T>(device_type);

  CALL_ALL_TYPES(REGISTER_KERNEL)
}