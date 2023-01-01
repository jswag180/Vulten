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

#include "shaders/headers/BiasAddGrad/BiasAddGrad.h"

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

struct BiasAddGradOp {
  BiasAddGradOp() : format("AAAA") {}
  std::string format;
};

void* BiasAddGradOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel = new BiasAddGradOp();

  StatusSafePtr status(TF_NewStatus());

  TF_OpKernelConstruction_GetAttrString(ctx, "data_format",
                                        kernel->format.data(), 4, status.get());

  return kernel;
}

void BiasAddGradOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<BiasAddGradOp*>(kernel);
  }
}

template <TF_DataType T, const std::vector<uint32_t>* spirv>
void BiasAddGradOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SCOPE_TIMER("BiasAddGradOp")

  BiasAddGradOp* biasAddGradOp = static_cast<BiasAddGradOp*>(kernel);

  StatusSafePtr status(TF_NewStatus());
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status.get());
  TensorSafePtr x_safe_ptr(x);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: BiasAddGradOp 1\n";
    return;
  }
  auto in_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(x_safe_ptr.get()));

  absl::InlinedVector<int64_t, 4> x_dims =
      absl::InlinedVector<int64_t, 4>(4, 1);
  for (auto i = 0; i < TF_NumDims(x_safe_ptr.get()); ++i) {
    x_dims[i + (4 - TF_NumDims(x_safe_ptr.get()))] =
        TF_Dim(x_safe_ptr.get(), i);
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0),
      &x_dims.data()[biasAddGradOp->format == "NHWC" ? 3 : 1], 1,
      x_dims.data()[biasAddGradOp->format == "NHWC" ? 3 : 1] *
          TF_DataTypeSize(T),
      status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    std::cout << "Error: BiasAddGradOp 2\n";
    return;
  }
  auto out_ptr = static_cast<std::shared_ptr<kp::TensorT<float>>*>(
      TF_TensorData(output_safe_ptr.get()));

  SP_Stream stream = TF_GetStream(ctx, status.get());
  MutexScopeLock guard = MutexScopeLock(&stream->instance->mainQueueMutex);

  uint32_t local_x =
      stream->instance->deviceProps[stream->deviceNum]
          .physicalProperties.limits.maxComputeWorkGroupInvocations;
  uint32_t subGroups =
      stream->instance->deviceProps[stream->deviceNum].maxSubGroupSize;

  uint32_t threads = 0;

  if (biasAddGradOp->format == "NHWC") {
    threads = uint32_t(std::ceil(
        uint32_t(in_ptr->get()->size() / x_dims.data()[3]) / float(local_x)));
  } else {
    threads = uint32_t(std::ceil(uint32_t(x_dims.data()[2] * x_dims.data()[3]) /
                                 float(local_x)));
  }

  std::vector<std::shared_ptr<kp::Sequence>> invocations(
      x_dims.data()[biasAddGradOp->format == "NHWC" ? 3 : 1]);

  for (uint32_t i = 0; i < invocations.size(); i++) {
    std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
        {*in_ptr, *out_ptr}, *spirv, kp::Workgroup({threads}),
        std::vector<uint32_t>{
            local_x, subGroups, biasAddGradOp->format == "NHWC" ? 0U : 1U,
            uint32_t(x_dims.data()[0]), uint32_t(x_dims.data()[1]),
            uint32_t(x_dims.data()[2]), uint32_t(x_dims.data()[3])},
        std::vector<uint32_t>{i});

    invocations[i] =
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(algo)
            ->evalAsync();
  }

  for (uint32_t i = 0; i < invocations.size(); i++) {
    invocations[i]->evalAwait();
  }
}

template <TF_DataType T, const std::vector<uint32_t>* spirv>
void RegisterBiasAddGradKernels(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder(
      "BiasAddGrad", device_type, BiasAddGradOp_Create,
      &BiasAddGradOp_Compute<T, spirv>, &BiasAddGradOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAddGrad kernel with attribute T";
  TF_RegisterKernelBuilder("BiasAddGrad", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering BiasAddGrad kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceBiasAddGradOp(const char* device_type) {
#define REGISTER_KERNEL(T, S)                                             \
  vulten_plugin::RegisterBiasAddGradKernels<T, &shader::BiasAddGrad_##S>( \
      device_type);

#ifdef BIASADDGRAD_FLOAT
  REGISTER_KERNEL(TF_FLOAT, float)
#endif
#ifdef BIASADDGRAD_INT
  REGISTER_KERNEL(TF_INT32, int)
#endif
#ifdef BIASADDGRAD_UINT
  REGISTER_KERNEL(TF_UINT32, uint)
#endif
#ifdef BIASADDGRAD_INT64_T
  REGISTER_KERNEL(TF_INT64, int64_t)
#endif
#ifdef BIASADDGRAD_UINT64_T
  REGISTER_KERNEL(TF_UINT64, uint64_t)
#endif
#ifdef BIASADDGRAD_INT8_T
  REGISTER_KERNEL(TF_INT8, int8_t)
#endif
#ifdef BIASADDGRAD_UINT8_T
  REGISTER_KERNEL(TF_UINT8, uint8_t)
#endif
#ifdef BIASADDGRAD_DOUBLE
  REGISTER_KERNEL(TF_DOUBLE, double)
#endif
}