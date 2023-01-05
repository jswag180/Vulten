#pragma once

#include "tensorflow/c/tf_tensor.h"

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
#define GET_INPUT_TENSOR(OP_NAME, NAME, NUM, CTX, STATUS)         \
  TF_Tensor* NAME = nullptr;                                      \
  TF_GetInput(CTX, NUM, &NAME, STATUS.get());                     \
  TensorSafePtr NAME##_safe_ptr(NAME);                            \
  if (TF_GetCode(STATUS.get()) != TF_OK) {                        \
    TF_OpKernelContext_Failure(CTX, STATUS.get());                \
    std::cout << "Error: " << OP_NAME << " at " << #NAME << "\n"; \
    return;                                                       \
  }                                                               \
  if (TF_TensorElementCount(NAME##_safe_ptr.get()) == 0) return;  \
  absl::InlinedVector<int64_t, 4> NAME##_dims(                    \
      TF_NumDims(NAME##_safe_ptr.get()));                         \
  for (auto i = 0; i < TF_NumDims(NAME##_safe_ptr.get()); ++i) {  \
    NAME##_dims[i] = TF_Dim(NAME##_safe_ptr.get(), i);            \
  }

#define MAKE_OUTPUT_TENSOR(OP_NAME, NAME, NUM, DIMS, TYPE, CTX, STATUS)        \
  TensorSafePtr output_safe_ptr(TF_AllocateOutput(                             \
      CTX, NUM, TF_ExpectedOutputDataType(CTX, NUM), DIMS.data(), DIMS.size(), \
      TF_TensorElementCount(input_safe_ptr.get()) * TF_DataTypeSize(TYPE),     \
      STATUS.get()));                                                          \
  if (TF_GetCode(STATUS.get()) != TF_OK) {                                     \
    TF_OpKernelContext_Failure(CTX, STATUS.get());                             \
    std::cout << "Error: " << OP_NAME << " at " << #NAME << "\n";              \
    return;                                                                    \
  }
