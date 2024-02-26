#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "Vulten_backend/ops/multiFunc/MultiFunc_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T, uint32_t OP>
void MultiFuncOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("MultiFuncOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor input = tensor_utills::get_input_tensor(
      "MultiFuncOp:input", 0, ctx, status.get());

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "MultiFuncOp:output", 0, input.dims, ctx, status.get());

  if (input.is_empty) {
    return;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::multiFunc::run_op(inst, (vulten_ops::Data_type)T,
                                input.vulten_tensor, output.vulten_tensor, OP);
}

template <TF_DataType T, uint32_t OP>
void RegisterMultiFuncOpKernel(const char* device_type) {
  std::string op = vulten_ops::multiFunc::op_as_str(OP);

  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(op.c_str(), device_type, nullptr,
                                      &MultiFuncOp_Compute<T, OP>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MultiFuncOp kernel with attribute T";
  TF_RegisterKernelBuilder(op.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering MultiFuncOp kernel";
}

void RegisterDeviceMultiFunc(const char* device_type) {
#define REGISTER_KERNEL(T)                              \
  RegisterMultiFuncOpKernel<T, OP_SQRT>(device_type);   \
  RegisterMultiFuncOpKernel<T, OP_EXP>(device_type);    \
  RegisterMultiFuncOpKernel<T, OP_LOG>(device_type);    \
  RegisterMultiFuncOpKernel<T, OP_SQUARE>(device_type); \
  RegisterMultiFuncOpKernel<T, OP_NEG>(device_type);

  // Sqrt is hard with compute shaders with the lack of native
  // pow, exp, cos, and sin in extended types.
  REGISTER_KERNEL(TF_FLOAT)
}