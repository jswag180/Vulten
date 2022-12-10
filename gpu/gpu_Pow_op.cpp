#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/Pow/Pow.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
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

namespace vulten_plugin {

template <TF_DataType T, const std::vector<uint32_t>* spirv>
void PowOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("PowOp")

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status.get());
  TensorSafePtr x_safe_ptr(x);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: pow 1\n";
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
    std::cout << "Error: pow 1\n";
    return;
  }
  if (TF_TensorElementCount(y_safe_ptr.get()) == 0) return;
  auto y_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(y_safe_ptr.get()));

  if (TF_TensorElementCount(x_safe_ptr.get()) !=
          TF_TensorElementCount(y_safe_ptr.get()) &&
      TF_TensorElementCount(x_safe_ptr.get()) != 1 &&
      TF_TensorElementCount(y_safe_ptr.get()) != 1) {
    std::cerr << "Error input size mismatch in Pow \n";
    exit(-1);
  }

  absl::InlinedVector<int64_t, 4> x_dims =
      absl::InlinedVector<int64_t, 4>(TF_NumDims(x_safe_ptr.get()));
  // std::cout << "X: \n";
  for (auto i = 0; i < TF_NumDims(x_safe_ptr.get()); ++i) {
    x_dims[i] = TF_Dim(x_safe_ptr.get(), i);
    // std::cout << TF_Dim(x_safe_ptr.get(), i) << "\n";
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), x_dims.data(), x_dims.size(),
      TF_TensorElementCount(x_safe_ptr.get()) * TF_DataTypeSize(T),
      status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: pow 2\n";
    return;
  }
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  uint32_t scaler = 0;
  if (TF_TensorElementCount(x_safe_ptr.get()) == 1) {
    scaler = 1;
  } else if (TF_TensorElementCount(y_safe_ptr.get()) == 1) {
    scaler = 2;
  }

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*x_ptr, *y_ptr, *out_ptr}, *spirv,
      kp::Workgroup({uint32_t(TF_TensorElementCount(x_safe_ptr.get()))}), {},
      std::vector<uint32_t>{scaler});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <TF_DataType T, const std::vector<uint32_t>* spirv>
void RegisterPowOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Pow", device_type, nullptr,
                                      &PowOp_Compute<T, spirv>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel with attribute T";
  TF_RegisterKernelBuilder("Pow", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel";
}

}  // namespace vulten_plugin

void RegisterDevicePow(const char* device_type) {
#define REGISTER_KERNEL(T, S) \
  vulten_plugin::RegisterPowOpKernel<T, &shader::Pow_##S>(device_type);

#ifdef POW_FLOAT
  REGISTER_KERNEL(TF_FLOAT, float)
#endif
#ifdef POW_INT
  REGISTER_KERNEL(TF_INT32, int)
#endif
#ifdef POW_UINT
  REGISTER_KERNEL(TF_UINT32, uint)
#endif
#ifdef POW_INT64_T
  REGISTER_KERNEL(TF_INT64, int64_t)
#endif
#ifdef POW_UINT64_T
  REGISTER_KERNEL(TF_UINT64, uint64_t)
#endif
#ifdef POW_INT8_T
  REGISTER_KERNEL(TF_INT8, int8_t)
#endif
#ifdef POW_UINT8_T
  REGISTER_KERNEL(TF_UINT8, uint8_t)
#endif
#ifdef POW_DOUBLE
  REGISTER_KERNEL(TF_DOUBLE, double)
#endif
}