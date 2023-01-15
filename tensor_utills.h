#pragma once

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor* t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

/**
 * This sets up a safe ptr to the underlying tf_tensor as NAME_safe_ptr.
 * Also makes a dims array called NAME_dims
 * @param OP_NAME The name of the op for error printing.
 * @param NAME Name to give the tensor.
 * @param NUM The input tensor index to get.
 */
#define GET_INPUT_TENSOR(OP_NAME, NAME, NUM, CTX, STATUS)          \
  TF_Tensor* NAME = nullptr;                                       \
  TF_GetInput(CTX, NUM, &NAME, STATUS.get());                      \
  TensorSafePtr NAME##_safe_ptr(NAME);                             \
  if (TF_GetCode(STATUS.get()) != TF_OK) {                         \
    TF_OpKernelContext_Failure(CTX, STATUS.get());                 \
    std::cout << "Error: " << OP_NAME << " at " << #NAME << "\n";  \
    return;                                                        \
  }                                                                \
  if (TF_TensorElementCount(NAME##_safe_ptr.get()) == 0) return;   \
  absl::InlinedVector<int64_t, 4> NAME##_dims(                     \
      TF_NumDims(NAME##_safe_ptr.get()));                          \
  for (auto i = 0; i < TF_NumDims(NAME##_safe_ptr.get()); ++i) {   \
    NAME##_dims[i] = TF_Dim(NAME##_safe_ptr.get(), i);             \
  }                                                                \
  vulten_ops::Vulten_tensor NAME##_tensor(                         \
      VOID_TO_DEVICE_BUFFER(TF_TensorData(NAME##_safe_ptr.get())), \
      NAME##_dims.size(), NAME##_dims.data());

#define GET_INPUT_FROM_VAR(NAME, NUM, CTX, LOCKING, STATUS)                   \
  std::unique_ptr<TF_Tensor*> NAME##_ref =                                    \
      std::unique_ptr<TF_Tensor*>(new TF_Tensor*);                            \
  TF_GetInputTensorFromVariable(CTX, NUM, LOCKING, 0, 0,                      \
                                &tensor_utills::copyFunc, NAME##_ref.get(),   \
                                STATUS);                                      \
  absl::InlinedVector<int64_t, 4> NAME##_dims(TF_NumDims(*NAME##_ref.get())); \
  for (auto i = 0; i < TF_NumDims(*NAME##_ref.get()); ++i) {                  \
    NAME##_dims[i] = TF_Dim(*NAME##_ref.get(), i);                            \
  }                                                                           \
  vulten_ops::Vulten_tensor NAME##_tensor(                                    \
      VOID_TO_DEVICE_BUFFER(TF_TensorData(*NAME##_ref.get())),                \
      NAME##_dims.size(), NAME##_dims.data());

#define MAKE_OUTPUT_TENSOR(OP_NAME, NAME, NUM, DIMS, TYPE, CTX, STATUS)        \
  uint64_t total_##NAME##_elements = 1;                                        \
  for (uint64_t i = 0; i < DIMS.size(); i++) {                                 \
    total_##NAME##_elements *= DIMS[i];                                        \
  }                                                                            \
  TensorSafePtr output_safe_ptr(TF_AllocateOutput(                             \
      CTX, NUM, TF_ExpectedOutputDataType(CTX, NUM), DIMS.data(), DIMS.size(), \
      total_##NAME##_elements* TF_ExpectedOutputDataType(CTX, NUM),            \
      STATUS.get()));                                                          \
  if (TF_GetCode(STATUS.get()) != TF_OK) {                                     \
    TF_OpKernelContext_Failure(CTX, STATUS.get());                             \
    std::cout << "Error: " << OP_NAME << " at " << #NAME << "\n";              \
    return;                                                                    \
  }                                                                            \
  vulten_ops::Vulten_tensor NAME##_tensor(                                     \
      VOID_TO_DEVICE_BUFFER(TF_TensorData(NAME##_safe_ptr.get())),             \
      DIMS.size(), DIMS.data());

namespace tensor_utills {

static void copyFunc(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest) {
  StatusSafePtr status(TF_NewStatus());
  SP_Stream stream = TF_GetStream(ctx, status.get());

  TensorSafePtr source_safe_ptr(source);
  auto source_buffer =
      VOID_TO_DEVICE_BUFFER(TF_TensorData(source_safe_ptr.get()));

  TensorSafePtr dest_safe_ptr(dest);
  auto dest_buffer = VOID_TO_DEVICE_BUFFER(TF_TensorData(dest_safe_ptr.get()));

  VOID_TO_INSTANCE(stream->instance)->copy_buffer(source_buffer, dest_buffer);
}

};  // namespace tensor_utills