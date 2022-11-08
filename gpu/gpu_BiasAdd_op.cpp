#include <iostream>

#include "absl/container/inlined_vector.h"
#include "gpuBackend.h"
#include "shaders/headers/shaderBiasAdd.hpp"
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

template <TF_DataType T>
struct BiasAddOp {
  BiasAddOp() : data_format_("") {}
  std::string data_format_;
  uint32_t channelIndx_;
};

template <TF_DataType T>
void* BiasAddOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new BiasAddOp<T>();

  StatusSafePtr status(TF_NewStatus());

  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx, "data_format", &list_size,
                                      &total_size, status.get());
  std::vector<char> format_vec(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format_vec.data(),
                                        total_size, status.get());
  kernel->data_format_ = std::move(std::string(format_vec.data(), total_size));

  if (kernel->data_format_ == "NHWC") {
    kernel->channelIndx_ = 3;
  } else if (kernel->data_format_ == "NCHW") {
    kernel->channelIndx_ = 1;
  } else {
    std::cerr << "Error: input data format: " << kernel->data_format_
              << " is not supported.\n";
  }

  return kernel;
}

template <TF_DataType T>
void BiasAddOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<BiasAddOp<T>*>(kernel);
  }
}

template <TF_DataType T>
void BiasAddOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  BiasAddOp<T>* biasAddOp = static_cast<BiasAddOp<T>*>(kernel);

  StatusSafePtr status(TF_NewStatus());

  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  TensorSafePtr input_safe_ptr(input);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_TensorElementCount(input_safe_ptr.get()) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(input_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(input_safe_ptr.get()); ++i) {
    dims[i] = TF_Dim(input_safe_ptr.get(), i);
  }

  TF_Tensor* bias = nullptr;
  TF_GetInput(ctx, 1, &bias, status.get());
  TensorSafePtr bias_safe_ptr(bias);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
      TF_TensorElementCount(input_safe_ptr.get()) * TF_DataTypeSize(T),
      status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }

  if (dims[biasAddOp->channelIndx_ == 1
               ? biasAddOp->channelIndx_
               : TF_NumDims(input_safe_ptr.get()) - 1] != TF_Dim(bias, 0)) {
    std::cerr << "Error: Input dim 4 ("
              << dims[TF_NumDims(input_safe_ptr.get()) - 1]
              << ") does not match bias (" << TF_Dim(bias, 0) << ")\n";
    exit(-1);
  }

  SP_Stream stream = TF_GetStream(ctx, status.get());
  // std::lock_guard<std::mutex> guard(stream->instance->testMutex);
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(input_safe_ptr.get()));
  auto bias_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(bias_safe_ptr.get()));
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
      {*in_ptr, *bias_ptr, *out_ptr}, spirv,
      kp::Workgroup({in_ptr->get()->size()}), {},
      std::vector<uint32_t>{
          uint32_t(dims[biasAddOp->channelIndx_ == 1
                            ? biasAddOp->channelIndx_
                            : TF_NumDims(input_safe_ptr.get()) - 1])});

  stream->instance->mngr->sequence(stream->instance->mainQueue)
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval();
}

template <TF_DataType T>
void RegisterBiasAddOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("BiasAdd", device_type, BiasAddOp_Create<T>,
                          &BiasAddOp_Compute<T>, &BiasAddOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering bias add kernel with attribute T";
  TF_RegisterKernelBuilder("BiasAddOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering bias add kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceBiasAdd(const char* device_type) {
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv,
                     kp::shader_data::___shaders_BiasAdd_comp_spv)

  vulten_plugin::RegisterBiasAddOpKernel<TF_FLOAT>(device_type);
}