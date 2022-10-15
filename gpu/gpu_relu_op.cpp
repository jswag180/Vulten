#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/shaderRelu.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

namespace vulten_plugin {

static std::vector<uint32_t> spirv;

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

// struct ReluOp {
//   float alpha_;
//   float beta_;
// };

// void* ReluOp_Create(ReluOp* kernel, float alpha, float beta) {
//   kernel->alpha_ = alpha;
//   kernel->beta_ = beta;
//   return kernel;
// }

template <typename T>
void ReluOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  // ReluOp* relu = static_cast<ReluOp*>(kernel);
  StatusSafePtr status(TF_NewStatus());
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  TensorSafePtr input_safe_ptr(input);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: relu 1\n";
    return;
  }
  if (TF_TensorElementCount(input_safe_ptr.get()) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(input_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(input_safe_ptr.get()); ++i) {
    dims[i] = TF_Dim(input_safe_ptr.get(), i);
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
      TF_TensorElementCount(input_safe_ptr.get()) * sizeof(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: relu 2\n";
    return;
  }

  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(input_safe_ptr.get()));
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*in_ptr, *out_ptr}, spirv,
      kp::Workgroup({(uint32_t)in_ptr->get()->size()}));

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <typename T>
void RegisterReluOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Relu", device_type, nullptr,
                                      &ReluOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel with attribute T";
  TF_RegisterKernelBuilder("ReluOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceRelu(const char* device_type) {
  vulten_plugin::spirv.resize(kp::shader_data::___shaders_Relu_comp_spv_len /
                              4);
  memcpy(&vulten_plugin::spirv[0], kp::shader_data::___shaders_Relu_comp_spv,
         kp::shader_data::___shaders_Relu_comp_spv_len);

  vulten_plugin::RegisterReluOpKernel<float>(device_type);
}