#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

// shaders
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "shaders/headers/shaderBasicOps.hpp"

#define OP_MUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_DIV 3
#define OP_DIV_NO_NAN 4

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

namespace vulten_plugin {

static std::vector<uint32_t> spirv_basic;

template <TF_DataType T, uint32_t OP>
void BasicOps_Compute(void* kernel, TF_OpKernelContext* ctx) {
  StatusSafePtr status(TF_NewStatus());
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status.get());
  TensorSafePtr x_safe_ptr(x);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: mul 1\n";
    return;
  }
  if (TF_TensorElementCount(x_safe_ptr.get()) == 0) return;
  auto x_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(x_safe_ptr.get()));

  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status.get());
  TensorSafePtr y_safe_ptr(y);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: mul 1\n";
    return;
  }
  if (TF_TensorElementCount(y_safe_ptr.get()) == 0) return;
  auto y_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(y_safe_ptr.get()));

  absl::InlinedVector<int64_t, 4> x_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(x_safe_ptr.get()); ++i) {
    x_dims[i + (4 - TF_NumDims(x_safe_ptr.get()))] =
        TF_Dim(x_safe_ptr.get(), i);
  }

  absl::InlinedVector<int64_t, 4> y_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(y_safe_ptr.get()); ++i) {
    y_dims[i + (4 - TF_NumDims(y_safe_ptr.get()))] =
        TF_Dim(y_safe_ptr.get(), i);
  }

  absl::InlinedVector<int64_t, 4> res_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  // std::cout << "x: " << TF_NumDims(x_safe_ptr.get()) << " y: " <<
  // TF_NumDims(y_safe_ptr.get()) << " res: " << res_dims.size() << "\n";
  int64_t resNumElements = 1;
  int64_t numOfResDims =
      std::max(TF_NumDims(x_safe_ptr.get()), TF_NumDims(y_safe_ptr.get()));
  for (int64_t i = 3; i >= 0; i--) {
    res_dims[i] = std::max(x_dims[i], y_dims[i]);
    resNumElements *= res_dims[i];
    // std::cout << x << " | " << y << " | " << res_dims[i] << "\n";
  }

  TensorSafePtr output_safe_ptr(
      TF_AllocateOutput(ctx, 0, TF_ExpectedOutputDataType(ctx, 0),
                        &res_dims.data()[4 - numOfResDims], numOfResDims,
                        resNumElements * TF_DataTypeSize(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: mul 2\n";
    return;
  }
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::vector<std::shared_ptr<kp::Sequence>> sequences(res_dims[0]);
  for (uint32_t i = 0; i < uint32_t(res_dims[0]); i++) {
    std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
        {*x_ptr, *y_ptr, *out_ptr}, spirv_basic,
        kp::Workgroup({uint32_t(res_dims[1]), uint32_t(res_dims[2]),
                       uint32_t(res_dims[3])}),
        std::vector<uint32_t>{
            i, uint32_t(res_dims[0]), uint32_t(x_dims[0]), uint32_t(x_dims[1]),
            uint32_t(x_dims[2]), uint32_t(x_dims[3]), uint32_t(y_dims[0]),
            uint32_t(y_dims[1]), uint32_t(y_dims[2]), uint32_t(y_dims[3]), OP},
        {});
    // stream->instance->mngr->sequence(stream->instance->mainQueue)->record<kp::OpAlgoDispatch>(algo)->eval();
    sequences[i] = stream->instance->mngr->sequence(stream->instance->mainQueue)
                       ->record<kp::OpAlgoDispatch>(algo)
                       ->evalAsync();
  }

  for (uint32_t i = 0; i < uint32_t(res_dims[0]); i++) {
    sequences[i]->evalAwait();
  }
}

template <TF_DataType T, uint32_t OP>
void RegisterBasicOpKernels(const char* device_type) {
  std::string op = "";
  if (OP == OP_MUL) {
    op = "Mul";
  } else if (OP == OP_ADD) {
    op == "Add";
  } else if (OP == OP_SUB) {
    op == "Sub";
  } else if (OP == OP_DIV) {
    op == "Div";
  } else if (OP == OP_DIV_NO_NAN) {
    op == "DivNoNan";
  }

  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(op.c_str(), device_type, nullptr,
                                      &BasicOps_Compute<T, OP>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Mul kernel with attribute T";
  TF_RegisterKernelBuilder(op.c_str(), builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Mul kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceBasicOps(const char* device_type) {
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv_basic,
                     kp::shader_data::___shaders_BasicOps_comp_spv)

#define REGISTER_KERNELS(T)                                      \
  vulten_plugin::RegisterBasicOpKernels<T, OP_MUL>(device_type); \
  vulten_plugin::RegisterBasicOpKernels<T, OP_ADD>(device_type); \
  vulten_plugin::RegisterBasicOpKernels<T, OP_SUB>(device_type); \
  vulten_plugin::RegisterBasicOpKernels<T, OP_DIV>(device_type); \
  vulten_plugin::RegisterBasicOpKernels<T, OP_DIV_NO_NAN>(device_type);

  REGISTER_KERNELS(TF_FLOAT)

#undef REGISTER_KERNELS
}
