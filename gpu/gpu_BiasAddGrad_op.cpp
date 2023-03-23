#include <iostream>

#include "Vulten_backend/ops/Sum_op.h"
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

  GET_INPUT_TENSOR("BiasAddGradOp", input, 0, ctx, status)

  std::vector<int32_t> axis_vec = std::vector<int32_t>();

  absl::InlinedVector<int64_t, 4> out_dims(1);
  if (biasAddGradOp_info->format == "NHWC") {
    out_dims[0] = input_dims[input_dims.size() - 1];
    for (uint32_t i = 0; i < input_dims.size() - 1; i++) {
      axis_vec.push_back(i);
    }
    std::reverse(axis_vec.begin(), axis_vec.end());
  } else if (biasAddGradOp_info->format == "NCHW") {
    out_dims[0] = input_dims[1];
    for (uint32_t i = 0; i < input_dims.size(); i++) {
      if (i != 1) {
        axis_vec.push_back(i);
      }
    }
    std::reverse(axis_vec.begin(), axis_vec.end());
  }

  MAKE_OUTPUT_TENSOR("BiasAddGradOp", output, 0, out_dims, T, ctx, status)

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::Sum_op* sum_op = nullptr;
  std::string op_cache_name = "Sum";
  inst->main_queue_mutex.lock();
  if (inst->op_chache.find(op_cache_name) == inst->op_chache.end()) {
    inst->op_chache[op_cache_name] =
        (vulten_ops::Vulten_op*)new vulten_ops::Sum_op(inst);
  }
  sum_op = (vulten_ops::Sum_op*)inst->op_chache[op_cache_name];
  inst->main_queue_mutex.unlock();

  sum_op->run_op((vulten_ops::Data_type)T, input_tensor, axis_vec,
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