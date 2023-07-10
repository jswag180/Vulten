#include "tensor_utills.h"

#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

namespace tensor_utills {

Input_tensor::~Input_tensor() { TF_DeleteTensor(tf_tensor); }

Input_tensor get_input_tensor(const char* name, int input_num,
                              TF_OpKernelContext* ctx, TF_Status* status) {
  Input_tensor input_tensor = Input_tensor();

  TF_Tensor* tf_tensor = nullptr;
  TF_GetInput(ctx, input_num, &tf_tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    std::cout << "Input Error at: " << name << "\n";
    exit(-1);
  }
  input_tensor.tf_tensor = tf_tensor;

  input_tensor.dims = absl::InlinedVector<int64_t, 4>(TF_NumDims(tf_tensor));
  for (auto i = 0; i < TF_NumDims(tf_tensor); ++i) {
    input_tensor.dims[i] = TF_Dim(tf_tensor, i);
  }

  input_tensor.vulten_tensor = vulten_ops::Vulten_tensor(
      VOID_TO_DEVICE_BUFFER(TF_TensorData(tf_tensor)), input_tensor.dims.size(),
      input_tensor.dims.data());

  input_tensor.is_scalar =
      input_tensor.dims.size() == 0 && TF_TensorElementCount(tf_tensor) == 1;
  input_tensor.is_empty = TF_TensorElementCount(tf_tensor) == 0;

  return input_tensor;
}

Input_tensor get_input_tensor_from_var(const char* name, int input_num,
                                       bool lock, TF_OpKernelContext* ctx,
                                       TF_Status* status) {
  Input_tensor input_tensor = Input_tensor();

  TF_Tensor** tf_tensor = nullptr;
  TF_GetInputTensorFromVariable(ctx, input_num, lock, 0, 0,
                                &tensor_utills::copyFunc, tf_tensor, status);
  input_tensor.tf_tensor = *tf_tensor;

  input_tensor.dims =
      absl::InlinedVector<int64_t, 4>(TF_NumDims(input_tensor.tf_tensor));
  for (auto i = 0; i < TF_NumDims(input_tensor.tf_tensor); ++i) {
    input_tensor.dims[i] = TF_Dim(input_tensor.tf_tensor, i);
  }

  input_tensor.vulten_tensor = vulten_ops::Vulten_tensor(
      VOID_TO_DEVICE_BUFFER(TF_TensorData(input_tensor.tf_tensor)),
      input_tensor.dims.size(), input_tensor.dims.data());

  input_tensor.is_scalar = input_tensor.dims.size() == 0 &&
                           TF_TensorElementCount(input_tensor.tf_tensor) == 1;
  input_tensor.is_empty = TF_TensorElementCount(input_tensor.tf_tensor) == 0;

  return input_tensor;
}

Input_host_tensor::~Input_host_tensor() { TF_DeleteTensor(tf_tensor); }

Input_host_tensor get_input_host_tensor(const char* name, int input_num,
                                        TF_OpKernelContext* ctx,
                                        TF_Status* status) {
  Input_host_tensor input_host_tensor = Input_host_tensor();

  TF_Tensor* tf_tensor = nullptr;
  TF_GetInput(ctx, input_num, &tf_tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    std::cout << "Input Error at: " << name << "\n";
    exit(-1);
  }
  input_host_tensor.tf_tensor = tf_tensor;

  input_host_tensor.dims =
      absl::InlinedVector<int64_t, 4>(TF_NumDims(tf_tensor));
  for (auto i = 0; i < TF_NumDims(tf_tensor); ++i) {
    input_host_tensor.dims[i] = TF_Dim(tf_tensor, i);
  }

  input_host_tensor.type = TF_TensorType(tf_tensor);

  input_host_tensor.data = TF_TensorData(tf_tensor);

  input_host_tensor.is_scalar = input_host_tensor.dims.size() == 0 &&
                                TF_TensorElementCount(tf_tensor) == 1;
  input_host_tensor.is_empty = TF_TensorElementCount(tf_tensor) == 0;

  return input_host_tensor;
}

Output_tensor::~Output_tensor() { TF_DeleteTensor(tf_tensor); }

Output_tensor make_output_tensor(const char* name, int output_num,
                                 absl::InlinedVector<int64_t, 4>& dims,
                                 TF_OpKernelContext* ctx, TF_Status* status) {
  Output_tensor output_tensor;

  uint64_t total_elements = 1;
  for (uint64_t i = 0; i < dims.size(); i++) {
    total_elements *= dims[i];
  }

  TF_Tensor* tf_tensor = TF_AllocateOutput(
      ctx, output_num, TF_ExpectedOutputDataType(ctx, output_num), dims.data(),
      dims.size(), total_elements * TF_ExpectedOutputDataType(ctx, output_num),
      status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    std::cout << "Output Error at: " << name << "\n";
    exit(-1);
  }
  output_tensor.tf_tensor = tf_tensor;

  output_tensor.vulten_tensor =
      vulten_ops::Vulten_tensor(VOID_TO_DEVICE_BUFFER(TF_TensorData(tf_tensor)),
                                dims.size(), dims.data());

  output_tensor.is_scalar =
      dims.size() == 0 && TF_TensorElementCount(tf_tensor) == 1;

  return output_tensor;
}

void copyFunc(TF_OpKernelContext* ctx, TF_Tensor* source, TF_Tensor* dest) {
  StatusSafePtr status(TF_NewStatus());
  SP_Stream stream = TF_GetStream(ctx, status.get());

  TensorSafePtr source_safe_ptr(source);
  auto source_buffer =
      VOID_TO_DEVICE_BUFFER(TF_TensorData(source_safe_ptr.get()));

  TensorSafePtr dest_safe_ptr(dest);
  auto dest_buffer = VOID_TO_DEVICE_BUFFER(TF_TensorData(dest_safe_ptr.get()));

  if (TF_TensorElementCount(source_safe_ptr.get()) <= 0 ||
      TF_TensorElementCount(dest_safe_ptr.get()) <= 0) {
    return;
  }

  VOID_TO_INSTANCE(stream->instance)->copy_buffer(source_buffer, dest_buffer);
}

}  // namespace tensor_utills