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

template <TF_DataType T>
void StridedSliceOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  std::cout << "Strided slice \n";
}

template <TF_DataType T>
void RegisterStridedSliceOpKernels(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("StridedSlice", device_type, nullptr,
                                      &StridedSliceOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout
        << " Error while registering StridedSlice kernel with attribute T";
  TF_RegisterKernelBuilder("StridedSlice", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering StridedSlice kernel";
}

}  // namespace vulten_plugin

void RegisterDeviceStridedSliceOp(const char* device_type) {
  // LOAD_SHADER_TO_VEC(vulten_plugin::spirv_basic,
  // kp::shader_data::___shaders_BasicOps_comp_spv)

#define REGISTER_KERNELS(T) \
  vulten_plugin::RegisterStridedSliceOpKernels<T>(device_type);

  REGISTER_KERNELS(TF_INT32)

#undef REGISTER_KERNELS
}