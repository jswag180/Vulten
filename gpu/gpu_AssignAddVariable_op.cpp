#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "gpu_variable_helpers.h"
//#include "shaders/headers/shaderAddInPlace.hpp"
#include "shaders/headers/AddInPlace/AddInPlace.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

namespace vulten_plugin {

template <TF_DataType T, const std::vector<uint32_t> *spirv>
void AssignAddVariableOp_Compte(void *kernel, TF_OpKernelContext *ctx) {
  // utills::ScopeTimer timer("AssignAddVariableOp");
  StatusSafePtr status(TF_NewStatus());

  TF_AssignUpdateVariable(
      ctx, 0, 1, 0, 0, &varHelpers::copyFunc,
      [](TF_OpKernelContext *ctx, TF_Tensor *tensor, TF_Tensor *value, int Op) {
        StatusSafePtr status(TF_NewStatus());
        SP_Stream stream = TF_GetStream(ctx, status.get());

        MutexScopeLock guard =
            MutexScopeLock(&stream->instance->mainQueueMutex);

        TensorSafePtr tensor_safe_ptr(tensor);
        auto tensor_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
            TF_TensorData(tensor_safe_ptr.get()));

        TensorSafePtr value_safe_ptr(value);
        auto value_ptr = static_cast<std::shared_ptr<kp::TensorT<float>> *>(
            TF_TensorData(value_safe_ptr.get()));

        std::shared_ptr<kp::Algorithm> algo = stream->instance->mngr->algorithm(
            {*tensor_ptr, *value_ptr}, *spirv,
            kp::Workgroup({tensor_ptr->get()->size()}));
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(algo)
            ->eval();
      },
      status.get());
}

template <TF_DataType T, const std::vector<uint32_t> *spirv>
void RegisterAssignAddVariableOp(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder =
      TF_NewKernelBuilder("AssignAddVariableOp", device_type, nullptr,
                          &AssignAddVariableOp_Compte<T, spirv>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", T, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout
        << " Error while registering AssignAddVariable kernel with attribute T";
  TF_RegisterKernelBuilder("AssignAddVariableOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering AssignAddVariable kernel";
}

}  // namespace vulten_plugin

void RegisterAssignAddVariable(const char *device_type) {
#define REGISTER_KERNEL(T, S) \
  vulten_plugin::RegisterAssignAddVariableOp<T, &S>(device_type);

#ifdef ADDINPLACE_FLOAT
  REGISTER_KERNEL(TF_FLOAT, shader::AddInPlace_float)
#endif
#ifdef ADDINPLACE_INT
  REGISTER_KERNEL(TF_INT32, shader::AddInPlace_int)
#endif
#ifdef ADDINPLACE_UINT
  REGISTER_KERNEL(TF_UINT32, shader::AddInPlace_uint)
#endif
#ifdef ADDINPLACE_INT64_T
  REGISTER_KERNEL(TF_INT64, shader::AddInPlace_int64_t)
#endif
#ifdef ADDINPLACE_UINT64_T
  REGISTER_KERNEL(TF_UINT64, shader::AddInPlace_uint64_t)
#endif
#ifdef ADDINPLACE_INT8_T
  REGISTER_KERNEL(TF_INT8, shader::AddInPlace_int8_t)
#endif
#ifdef ADDINPLACE_UINT8_T
  REGISTER_KERNEL(TF_UINT8, shader::AddInPlace_uint8_t)
#endif
#ifdef ADDINPLACE_DOUBLE
  REGISTER_KERNEL(TF_DOUBLE, shader::AddInPlace_double)
#endif

#undef REGISTER_KERNEL
}
