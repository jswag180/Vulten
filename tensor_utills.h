#pragma once

#include <memory>

#include "Vulten_backend/ops/Vulten_backend_ops.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
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

namespace tensor_utills {

struct Input_tensor {
  TF_Tensor* tf_tensor;
  absl::InlinedVector<int64_t, 4> dims;
  vulten_ops::Vulten_tensor vulten_tensor;
  bool is_scalar;
  bool is_empty;

  ~Input_tensor();
};

struct Input_host_tensor {
  TF_Tensor* tf_tensor;
  absl::InlinedVector<int64_t, 4> dims;
  TF_DataType type;
  void* data;
  bool is_scalar;
  bool is_empty;

  ~Input_host_tensor();
};

struct Output_tensor {
  TF_Tensor* tf_tensor;
  vulten_ops::Vulten_tensor vulten_tensor;
  bool is_scalar;

  ~Output_tensor();
};

Input_tensor get_input_tensor(const char* name, int input_num,
                              TF_OpKernelContext* ctx, TF_Status* status);
Input_tensor get_input_tensor_from_var(const char* name, int input_num,
                                       bool lock, TF_OpKernelContext* ctx,
                                       TF_Status* status);
Input_host_tensor get_input_host_tensor(const char* name, int input_num,
                                        TF_OpKernelContext* ctx,
                                        TF_Status* status);
Output_tensor make_output_tensor(const char* name, int output_num,
                                 absl::InlinedVector<int64_t, 4>& dims,
                                 TF_OpKernelContext* ctx, TF_Status* status);

void copyFunc(TF_OpKernelContext* ctx, TF_Tensor* source, TF_Tensor* dest);

};  // namespace tensor_utills