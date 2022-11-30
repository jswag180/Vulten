#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "vulten_device.h"

// shaders
#include <iostream>
#include <memory>
#include <vector>

#include "shaders/headers/BatchAdd/BatchAdd.h"
#include "shaders/headers/Exp/Exp.h"
#include "shaders/headers/Softmax/Softmax.h"

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

template <TF_DataType T, const std::vector<uint32_t>* spirv_softmax,
          const std::vector<uint32_t>* spirv_batch_add,
          const std::vector<uint32_t>* spirv_exp>
void SoftmaxOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  StatusSafePtr status(TF_NewStatus());
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  TensorSafePtr input_safe_ptr(input);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: softmax 1\n";
    return;
  }

  if (TF_TensorElementCount(input_safe_ptr.get()) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(input_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(input_safe_ptr.get()); ++i) {
    dims[i] = TF_Dim(input_safe_ptr.get(), i);
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
      TF_TensorElementCount(input_safe_ptr.get()) * TF_DataTypeSize(T),
      status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: softmax 2\n";
    return;
  }

  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(input_safe_ptr.get()));
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  std::vector<float> stageVecExp(in_ptr->get()->size());
  auto expTen = stream->instance->mngr->tensorT<float>(
      {stageVecExp}, kp::Tensor::TensorTypes::eDevice);
  std::vector<float> stageVecBatchAdd(in_ptr->get()->size() / dims[1]);
  auto sumTen = stream->instance->mngr->tensorT<float>(
      {stageVecBatchAdd}, kp::Tensor::TensorTypes::eDevice);

  std::shared_ptr<kp::Algorithm> algo_exp = stream->instance->mngr->algorithm(
      {*in_ptr, expTen}, *spirv_exp,
      kp::Workgroup({(uint32_t)in_ptr->get()->size()}));
  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo_exp)
      ->eval();

  std::shared_ptr<kp::Algorithm> algo_batch_add =
      stream->instance->mngr->algorithm(
          {expTen, sumTen}, *spirv_batch_add,
          kp::Workgroup({(uint32_t)(in_ptr->get()->size() / dims[1])}), {},
          std::vector<uint32_t>{(uint32_t)dims[1]});
  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo_batch_add)
      ->eval();

  std::shared_ptr<kp::Algorithm> algo_softmax =
      stream->instance->mngr->algorithm(
          {expTen, sumTen, *out_ptr}, *spirv_softmax,
          kp::Workgroup({(uint32_t)in_ptr->get()->size()}), {},
          std::vector<uint32_t>{(uint32_t)dims[1]});
  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo_softmax)
      ->eval();
}

template <TF_DataType T, const std::vector<uint32_t>* spirv_softmax,
          const std::vector<uint32_t>* spirv_batch_add,
          const std::vector<uint32_t>* spirv_exp>
void RegisterSoftmaxOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(
      "Softmax", device_type, nullptr,
      &SoftmaxOp_Compute<T, spirv_softmax, spirv_batch_add, spirv_exp>,
      nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Softmax kernel with attribute T";
  TF_RegisterKernelBuilder("Softmax", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering Softmax kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceSoftmax(const char* device_type) {
#define REGISTER_KERNEL(T, S)                                            \
  vulten_plugin::RegisterSoftmaxOpKernel<                                \
      T, &shader::Softmax_##S, &shader::BatchAdd_##S, &shader::Exp_##S>( \
      device_type);

#ifdef SOFTMAX_FLOAT
  REGISTER_KERNEL(TF_FLOAT, float)
#endif
#ifdef SOFTMAX_INT
  REGISTER_KERNEL(TF_INT32, int)
#endif
#ifdef SOFTMAX_UINT
  REGISTER_KERNEL(TF_UINT32, uint)
#endif
#ifdef SOFTMAX_INT64_T
  REGISTER_KERNEL(TF_INT64, int64_t)
#endif
#ifdef SOFTMAX_UINT64_T
  REGISTER_KERNEL(TF_UINT64, uint64_t)
#endif
#ifdef SOFTMAX_INT8_T
  REGISTER_KERNEL(TF_INT8, int8_t)
#endif
#ifdef SOFTMAX_UINT8_T
  REGISTER_KERNEL(TF_UINT8, uint8_t)
#endif
#ifdef SOFTMAX_DOUBLE
  REGISTER_KERNEL(TF_DOUBLE, double)
#endif
}