#include <cstring>
#include <iostream>

#include "Vulten_backend/ops/reduce/Reduce_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

struct BiasAddGradOp {
  char format[5];
};

void* BiasAddGradOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new BiasAddGradOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", kernel->format, 5,
                                        status.get());

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

  tensor_utills::Input_tensor input = tensor_utills::get_input_tensor(
      "BiasAddGradOp:input", 0, ctx, status.get());

  std::vector<int32_t> axis_vec = std::vector<int32_t>();

  absl::InlinedVector<int64_t, 4> out_dims(1);
  if (std::strcmp(biasAddGradOp_info->format, "NHWC") == 0) {
    out_dims[0] = input.dims[input.dims.size() - 1];
    for (uint32_t i = 0; i < input.dims.size() - 1; i++) {
      axis_vec.push_back(i);
    }
    std::reverse(axis_vec.begin(), axis_vec.end());
  } else if (std::strcmp(biasAddGradOp_info->format, "NCHW") == 0) {
    out_dims[0] = input.dims[1];
    for (uint32_t i = 0; i < input.dims.size(); i++) {
      if (i != 1) {
        axis_vec.push_back(i);
      }
    }
    std::reverse(axis_vec.begin(), axis_vec.end());
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "BiasAddGradOp:output", 0, out_dims, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  if (input.is_empty) {
    inst->fill_buffer(output.vulten_tensor.buffer, 0, VK_WHOLE_SIZE, 0);
    return;
  }

  vulten_ops::reduce::run_op(inst, (vulten_ops::Data_type)T,
                             input.vulten_tensor, axis_vec,
                             output.vulten_tensor, OP_SUM);
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