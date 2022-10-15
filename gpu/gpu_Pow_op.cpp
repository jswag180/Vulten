#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

// shaders
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "shaders/headers/shaderPow.hpp"

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

static std::vector<uint32_t> spirv_pow;

template <typename T>
void PowOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
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
      TF_TensorElementCount(y_safe_ptr.get())) {
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

  // absl::InlinedVector<int64_t, 4> y_dims = absl::InlinedVector<int64_t,
  // 4>(TF_NumDims(y_safe_ptr.get())); std::cout << "Y: \n"; for (auto i = 0; i
  // < TF_NumDims(y_safe_ptr.get()); ++i) {
  //     x_dims[i] = TF_Dim(y_safe_ptr.get(), i);
  //     std::cout << TF_Dim(y_safe_ptr.get(), i) << "\n";
  // }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), x_dims.data(), x_dims.size(),
      TF_TensorElementCount(x_safe_ptr.get()) * sizeof(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: pow 2\n";
    return;
  }
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*x_ptr, *y_ptr, *out_ptr}, spirv_pow,
      kp::Workgroup({uint32_t(TF_TensorElementCount(x_safe_ptr.get()))}));

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <typename T>
void RegisterPowOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Pow", device_type, nullptr,
                                      &PowOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel with attribute T";
  TF_RegisterKernelBuilder("Pow", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Pow kernel";
}

}  // namespace vulten_plugin

void RegisterDevicePow(const char* device_type) {
  vulten_plugin::spirv_pow.resize(kp::shader_data::___shaders_Pow_comp_spv_len /
                                  4);
  memcpy(&vulten_plugin::spirv_pow[0], kp::shader_data::___shaders_Pow_comp_spv,
         kp::shader_data::___shaders_Pow_comp_spv_len);

  vulten_plugin::RegisterPowOpKernel<float>(device_type);
}