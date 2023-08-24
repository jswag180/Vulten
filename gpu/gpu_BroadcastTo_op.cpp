#include "Vulten_backend/ops/broadcast/Broadcast_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

template <TF_DataType T>
void BroadcastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BroadcastOp")

  StatusSafePtr status(TF_NewStatus());

  tensor_utills::Input_tensor input = tensor_utills::get_input_tensor(
      "BroadcastOp:input", 0, ctx, status.get());
  if (input.is_empty) {
    return;
  }

  tensor_utills::Input_host_tensor shape = tensor_utills::get_input_host_tensor(
      "BroadcastOp:shape", 1, ctx, status.get());
  if (shape.is_empty) {
    return;
  }

  if (input.dims.size() != TF_TensorElementCount(shape.tf_tensor)) {
    std::vector<int64_t> pad = std::vector<int64_t>(
        TF_TensorElementCount(shape.tf_tensor) - input.dims.size(), 1);
    input.dims.insert(input.dims.begin(), pad.begin(), pad.end());
    input.vulten_tensor.num_dims = input.dims.size();
  }

  absl::InlinedVector<int64_t, 4> res_dims =
      absl::InlinedVector<int64_t, 4>(input.dims.size(), 0);
  for (int64_t i = 0; i < res_dims.size(); i++) {
    if (shape.type == TF_INT32) {
      res_dims[i] = std::max(input.dims[i], int64_t(((int32_t*)shape.data)[i]));
    } else if (shape.type == TF_INT64) {
      res_dims[i] = std::max(input.dims[i], ((int64_t*)shape.data)[i]);
    }
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "BasicOp:output", 0, res_dims, ctx, status.get());

  if (output.is_scalar) {
    res_dims.resize(1, 1);
    output.vulten_tensor.num_dims = 1;
  }

  if (input.is_scalar) {
    input.dims.resize(1, 1);
    input.vulten_tensor.num_dims = 1;
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  vulten_ops::broadcast::run_op(inst, (vulten_ops::Data_type)T,
                                input.vulten_tensor, output.vulten_tensor);
}

template <TF_DataType T>
void RegisterBroadcastOpKernels(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("BroadcastTo", device_type, nullptr,
                                      &BroadcastOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BroadcastOp kernel with attribute T";

  TF_KernelBuilder_HostMemory(builder, "shape");

  TF_RegisterKernelBuilder("BroadcastTo", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BroadcastOp kernel";
}

void RegisterDeviceBroadcastOp(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterBroadcastOpKernels<T>(device_type);

  CALL_ALL_TYPES(REGISTER_KERNEL)
}