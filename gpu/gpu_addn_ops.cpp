#include <memory>

#include "Vulten_backend/ops/addn/Addn_op.h"
#include "absl/container/inlined_vector.h"
#include "scope_timer.h"
#include "tensor_utills.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

template <TF_DataType T>
void AddnOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("AddnOp")

  StatusSafePtr status(TF_NewStatus());

  int num_tensors = TF_NumInputs(ctx);

  tensor_utills::Input_tensor input =
      tensor_utills::get_input_tensor("AddnOp:input", 0, ctx, status.get());

  std::vector<vulten_ops::Vulten_tensor> tensors(num_tensors);

  tensors[0] = input.vulten_tensor;
  for (int i = 1; i < num_tensors; i++) {
    TF_Tensor* tensor_ptr = nullptr;
    TF_GetInput(ctx, i, &tensor_ptr, status.get());
    std::shared_ptr<TF_Tensor> tensor_safe_ptr(tensor_ptr, TensorDeleter{});
    tensors[i] = vulten_ops::Vulten_tensor(
        VOID_TO_DEVICE_BUFFER(TF_TensorData(tensor_safe_ptr.get())),
        input.dims.size(), input.dims.data());
  }

  tensor_utills::Output_tensor output = tensor_utills::make_output_tensor(
      "AddnOp:output", 0, input.dims, ctx, status.get());

  SP_Stream stream = TF_GetStream(ctx, status.get());
  vulten_backend::Instance* inst = stream->instance;

  if (num_tensors == 1) {
    inst->copy_buffer(input.vulten_tensor.buffer, output.vulten_tensor.buffer);
    return;
  }

  vulten_ops::addn::run_op(inst, (vulten_ops::Data_type)T, tensors,
                           output.vulten_tensor);
}

template <TF_DataType T>
void RegisterAddnOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("AddN", device_type, nullptr,
                                      &AddnOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sum kernel with attribute T";

  TF_RegisterKernelBuilder("AddN", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Sum kernel";
}

void RegisterDeviceAddn(const char* device_type) {
#define REGISTER_KERNEL(T) RegisterAddnOpKernel<T>(device_type);

  CALL_ALL_TYPES(REGISTER_KERNEL)
}