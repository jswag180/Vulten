#include <math.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpuBackend.h"
#include "gpu_variable_helpers.h"
#include "shaders/headers/shaderAddInPlace.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "vulten_device.h"

namespace vulten_plugin {

static std::vector<uint32_t> spirv;

template <TF_DataType T>
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
            {*tensor_ptr, *value_ptr}, spirv,
            kp::Workgroup({tensor_ptr->get()->size()}));
        stream->instance->mngr->sequence(stream->instance->mainQueue)
            ->record<kp::OpAlgoDispatch>(algo)
            ->eval();
      },
      status.get());
}

template <TF_DataType T>
void RegisterAssignAddVariableOp(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder =
      TF_NewKernelBuilder("AssignAddVariableOp", device_type, nullptr,
                          &AssignAddVariableOp_Compte<T>, nullptr);
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
  LOAD_SHADER_TO_VEC(vulten_plugin::spirv,
                     kp::shader_data::___shaders_AddInPlace_comp_spv)

  vulten_plugin::RegisterAssignAddVariableOp<TF_FLOAT>(device_type);
}
